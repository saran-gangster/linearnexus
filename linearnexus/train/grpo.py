"""Group Relative Policy Optimization (GRPO) Trainer.

GRPO is a simplified RL algorithm that:
1. Generates multiple completions per prompt
2. Scores them with reward function(s)
3. Uses PPO-style clipped policy gradient with group-normalized advantages
4. Optionally penalizes KL divergence from reference model

Key features:
- No value network required (uses group rewards for baseline)
- PPO-style clipping for stability
- Micro-batching for memory efficiency
- Optional KL penalty from reference model

Reference: https://github.com/joey00072/nanoGRPO
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from linearnexus.train.base import GRPOConfig
from linearnexus.checkpoint import CheckpointManager
from linearnexus.generate import generate

Array = jax.Array


class GRPOTrainer:
    """Group Relative Policy Optimization trainer.
    
    GRPO is a simplified RL algorithm that:
    1. Generates multiple completions per prompt
    2. Scores them with reward function(s)
    3. Uses PPO-style clipped policy gradient with group-normalized advantages
    4. Optionally penalizes KL divergence from reference model
    
    Based on: https://github.com/joey00072/nanoGRPO
    
    Key features:
    - No value network required (uses group rewards for baseline)
    - PPO-style clipping for stability
    - Micro-batching for memory efficiency
    - Optional KL penalty from reference model
    
    Args:
        model: Policy model to train.
        ref_model: Reference model for KL computation (frozen). 
                   If None, uses model with frozen old log probs only.
        reward_fns: List of reward functions (prompt_tokens, completion_tokens) -> reward.
        optimizer: Optax optimizer.
        config: GRPO configuration.
    
    Example:
        def length_reward(prompts, completions):
            # Reward longer completions
            return jnp.array([len(c) for c in completions]) / 100.0
        
        config = GRPOConfig(
            group_size=8,
            epsilon=0.1,
            max_gen_tokens=256,
        )
        trainer = GRPOTrainer(
            model=model,
            ref_model=None,  # No separate reference
            reward_fns=[length_reward],
            optimizer=optimizer,
            config=config,
        )
        trainer.train(prompt_dataloader)
    """
    
    def __init__(
        self,
        model: "LMModel",
        ref_model: Optional["LMModel"],
        reward_fns: List[Callable[[Array, Array], float]],
        optimizer: optax.GradientTransformation,
        config: GRPOConfig,
    ):
        from linearnexus.models import LMModel
        
        self.model = model
        self.ref_model = ref_model if ref_model is not None else model
        self.using_same_model = ref_model is None  # If True, ref = policy (no separate ref)
        self.reward_fns = reward_fns
        self.optimizer = optimizer
        self.config = config
        
        # Initialize optimizer
        self.opt_state = nnx.Optimizer(model, optimizer, wrt=nnx.Param)
        
        self.step = 0
        self.metrics_history = []
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_per_token_log_probs(
        self,
        model: "LMModel",
        input_ids: Array,
    ) -> Array:
        """Compute per-token log probabilities.
        
        Args:
            model: Language model.
            input_ids: Input token IDs [batch, seq].
            
        Returns:
            Log probabilities [batch, seq-1] (shifted).
        """
        logits, _ = model(input_ids)
        # Shift: predict position t+1 from position t
        logits = logits[:, :-1, :]  # [batch, seq-1, vocab]
        targets = input_ids[:, 1:]   # [batch, seq-1]
        
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Gather log probs for actual tokens
        token_log_probs = jnp.take_along_axis(
            log_probs, 
            targets[..., None], 
            axis=-1
        ).squeeze(-1)
        
        return token_log_probs
    
    def compute_rewards(
        self,
        prompts: Array,
        completions: Array,
    ) -> Array:
        """Compute rewards for completions using reward functions.
        
        Args:
            prompts: Prompt tokens [batch, prompt_len].
            completions: Completion tokens [batch, comp_len].
            
        Returns:
            Total rewards [batch].
        """
        batch_size = prompts.shape[0]
        total_rewards = jnp.zeros(batch_size)
        
        for reward_fn in self.reward_fns:
            # Reward function should handle batched inputs
            rewards = reward_fn(prompts, completions)
            if isinstance(rewards, (int, float)):
                rewards = jnp.full(batch_size, rewards)
            total_rewards = total_rewards + jnp.array(rewards)
        
        return total_rewards
    
    def compute_grpo_loss(
        self,
        input_ids: Array,
        old_policy_log_probs: Array,
        rewards: Array,
        mean_rewards: Array,
        std_rewards: Array,
        loss_mask: Array,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Compute GRPO loss with PPO-style clipping.
        
        Loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + beta * KL
        
        Where:
        - ratio = exp(log_pi - log_pi_old)
        - A = (reward - mean) / (std + eps)  [advantage]
        - KL = exp(log_ref - log_pi) - (log_ref - log_pi) - 1  [reverse KL approx]
        
        Args:
            input_ids: Full sequence (prompt + completion) [batch, seq].
            old_policy_log_probs: Log probs from policy before update [batch, seq-1].
            rewards: Reward scores [batch].
            mean_rewards: Mean reward in group [1] or [batch].
            std_rewards: Std of rewards in group [1] or [batch].
            loss_mask: Mask for completion tokens only [batch, seq-1].
            
        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Current policy log probs
        policy_log_probs = self.get_per_token_log_probs(self.model, input_ids)
        
        # Reference model log probs (for KL penalty)
        if self.config.beta > 0:
            ref_log_probs = self.get_per_token_log_probs(self.ref_model, input_ids)
            # Reverse KL approximation: exp(log_q - log_p) - (log_q - log_p) - 1
            log_ratios = ref_log_probs - policy_log_probs
            kl_div = jnp.exp(log_ratios) - log_ratios - 1
        else:
            kl_div = jnp.zeros_like(policy_log_probs)
            ref_log_probs = policy_log_probs
        
        # Advantage: normalized reward
        advantage = (rewards - mean_rewards) / (std_rewards + 1e-6)
        advantage = advantage.reshape(-1, 1)  # [batch, 1] for broadcasting
        
        # Policy ratio (importance sampling)
        policy_ratio = jnp.exp(policy_log_probs - old_policy_log_probs)
        
        # PPO-style clipped loss
        loss1 = policy_ratio * advantage
        loss2 = jnp.clip(
            policy_ratio, 
            1 - self.config.epsilon, 
            1 + self.config.epsilon
        ) * advantage
        
        # Negative because we want to maximize (gradient ascent)
        policy_loss = -jnp.minimum(loss1, loss2)
        
        # Apply mask and average over sequence
        policy_loss = (policy_loss * loss_mask).sum(axis=-1) / (loss_mask.sum(axis=-1) + 1e-6)
        kl_loss = (kl_div * loss_mask).sum(axis=-1) / (loss_mask.sum(axis=-1) + 1e-6)
        
        # Total loss: policy loss + KL penalty
        total_loss = policy_loss + self.config.beta * kl_loss
        
        metrics = {
            "loss": jnp.mean(total_loss),
            "policy_loss": jnp.mean(policy_loss),
            "kl": jnp.mean(kl_loss),
            "reward_mean": jnp.mean(rewards),
            "reward_std": jnp.std(rewards),
            "advantage_mean": jnp.mean(advantage),
        }
        
        return jnp.mean(total_loss), metrics
    
    def sample_and_score(
        self,
        prompts: Array,
        key: Array,
    ) -> Tuple[Array, Array, Array]:
        """Generate completions and compute rewards.
        
        Args:
            prompts: Prompt tokens [batch, prompt_len].
            key: Random key for generation.
            
        Returns:
            Tuple of (full_sequences, rewards, loss_mask).
        """
        batch_size, prompt_len = prompts.shape
        group_size = self.config.group_size
        
        # Expand prompts for group sampling
        prompts_expanded = jnp.repeat(prompts, group_size, axis=0)
        
        # Generate completions
        full_sequences = generate(
            self.model,
            prompts_expanded,
            max_tokens=self.config.max_gen_tokens,
            temperature=self.config.temperature,
            key=key,
        )
        
        # Extract completions
        completions = full_sequences[:, prompt_len:]
        
        # Compute rewards
        rewards = self.compute_rewards(prompts_expanded, completions)
        
        # Create loss mask (1 for completion tokens, 0 for prompt and padding)
        seq_len = full_sequences.shape[1]
        loss_mask = jnp.zeros((batch_size * group_size, seq_len - 1))
        
        # Mask completion tokens only (shifted by 1 for log probs alignment)
        completion_mask = jnp.ones((batch_size * group_size, completions.shape[1]))
        loss_mask = loss_mask.at[:, prompt_len-1:prompt_len-1+completions.shape[1]].set(completion_mask)
        
        # Also mask out padding tokens (assuming pad_token_id = 0)
        # TODO: Get actual pad token from tokenizer
        padding_mask = (full_sequences[:, 1:] != 0).astype(jnp.float32)
        loss_mask = loss_mask * padding_mask
        
        return full_sequences, rewards, loss_mask
    
    def train_step(
        self,
        prompts: Array,
        key: Array,
    ) -> Dict[str, Array]:
        """Single GRPO training step.
        
        1. Generate completions for each prompt (group_size per prompt)
        2. Score with reward function
        3. Compute old policy log probs
        4. Micro-batch: compute loss and gradients
        5. Update model
        
        Args:
            prompts: Prompt tokens [batch, prompt_len].
            key: Random key.
            
        Returns:
            Metrics dict.
        """
        batch_size = prompts.shape[0]
        group_size = self.config.group_size
        micro_group_size = self.config.micro_group_size
        
        # Step 1: Generate completions and get rewards
        key, gen_key = jax.random.split(key)
        full_sequences, rewards, loss_mask = self.sample_and_score(prompts, gen_key)
        
        # Reshape for batch processing: [batch, group, seq] 
        full_sequences = full_sequences.reshape(batch_size, group_size, -1)
        rewards = rewards.reshape(batch_size, group_size)
        loss_mask = loss_mask.reshape(batch_size, group_size, -1)
        
        all_metrics = []
        
        # Process each batch item's group
        for b in range(batch_size):
            b_sequences = full_sequences[b]  # [group_size, seq]
            b_rewards = rewards[b]           # [group_size]
            b_loss_mask = loss_mask[b]       # [group_size, seq-1]
            
            # Compute group statistics for advantage normalization
            mean_rewards = jnp.mean(b_rewards)
            std_rewards = jnp.std(b_rewards)
            
            # Step 2: Compute old policy log probs (before any updates)
            old_log_probs = self.get_per_token_log_probs(self.model, b_sequences)
            
            # Step 3: Micro-batch for memory efficiency
            num_micro_batches = group_size // micro_group_size
            micro_losses = []
            
            for m in range(num_micro_batches):
                start = m * micro_group_size
                end = start + micro_group_size
                
                m_sequences = b_sequences[start:end]
                m_old_log_probs = old_log_probs[start:end]
                m_rewards = b_rewards[start:end]
                m_loss_mask = b_loss_mask[start:end]
                
                # Compute loss and gradients
                def loss_fn(model):
                    # Temporarily set self.model for the loss computation
                    orig_model = self.model
                    self.model = model
                    loss, metrics = self.compute_grpo_loss(
                        m_sequences,
                        m_old_log_probs,
                        m_rewards,
                        mean_rewards,
                        std_rewards,
                        m_loss_mask,
                    )
                    self.model = orig_model
                    return loss
                
                loss, grads = nnx.value_and_grad(loss_fn)(self.model)
                
                # Accumulate gradients
                self.opt_state.update(self.model, grads)
                micro_losses.append(float(loss))
            
            # Compute metrics for this batch item
            avg_loss = sum(micro_losses) / len(micro_losses)
            all_metrics.append({
                "loss": avg_loss,
                "reward_mean": float(jnp.mean(b_rewards)),
                "reward_std": float(jnp.std(b_rewards)),
            })
        
        # Aggregate metrics
        final_metrics = {
            "loss": sum(m["loss"] for m in all_metrics) / len(all_metrics),
            "reward_mean": sum(m["reward_mean"] for m in all_metrics) / len(all_metrics),
            "reward_std": sum(m["reward_std"] for m in all_metrics) / len(all_metrics),
        }
        
        return final_metrics
    
    def train(
        self,
        prompt_dataloader: Iterator[Array],
        key: Optional[Array] = None,
    ) -> Dict[str, List[float]]:
        """Run GRPO training loop.
        
        Args:
            prompt_dataloader: Iterator yielding prompt token batches.
            key: Random key (uses config seed if None).
            
        Returns:
            Dict of metric histories.
        """
        if key is None:
            key = jax.random.PRNGKey(self.config.seed)
        
        print(f"Starting GRPO training for {self.config.max_steps} steps...")
        print(f"  Group size: {self.config.group_size}")
        print(f"  Micro group size: {self.config.micro_group_size}")
        print(f"  Beta (KL coef): {self.config.beta}")
        print(f"  Epsilon (clip): {self.config.epsilon}")
        
        start_time = time.time()
        
        for step in range(self.step, self.config.max_steps):
            # Get batch of prompts
            try:
                prompts = next(prompt_dataloader)
            except StopIteration:
                prompt_dataloader = iter(prompt_dataloader)
                prompts = next(prompt_dataloader)
            
            # Training step
            key, step_key = jax.random.split(key)
            metrics = self.train_step(prompts, step_key)
            
            self.step = step + 1
            self.metrics_history.append({"step": self.step, **metrics})
            
            # Logging
            if self.step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Step {self.step}/{self.config.max_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Reward: {metrics['reward_mean']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # Checkpointing
            if self.step % self.config.save_interval == 0:
                ckpt_manager = CheckpointManager(
                    self.output_dir,
                    max_to_keep=5,
                    best_metric="reward_mean",
                    best_mode="max",
                )
                ckpt_manager.save(
                    step=self.step,
                    model=self.model,
                    metrics=metrics,
                )
        
        print(f"GRPO training complete!")
        print(f"Final reward: {metrics['reward_mean']:.4f}")
        return {"metrics": self.metrics_history}
