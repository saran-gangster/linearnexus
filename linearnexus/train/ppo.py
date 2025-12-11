"""Proximal Policy Optimization (PPO) Trainer.

Full PPO with:
- Value network for advantage estimation
- GAE (Generalized Advantage Estimation)
- Policy and value clipping
- Entropy bonus

Note: This is a simplified implementation. Production PPO would include
more careful handling of episode boundaries, reward normalization, etc.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from linearnexus.train.base import (
    PPOConfig,
    compute_log_probs,
    kl_divergence,
)
from linearnexus.checkpoint import CheckpointManager

Array = jax.Array


class PPOTrainer:
    """Proximal Policy Optimization trainer.
    
    Full PPO with:
    - Value network for advantage estimation
    - GAE (Generalized Advantage Estimation)
    - Policy and value clipping
    - Entropy bonus
    
    Note: This is a simplified implementation. Production PPO would include
    more careful handling of episode boundaries, reward normalization, etc.
    
    Args:
        model: Policy model.
        value_model: Value network.
        ref_model: Reference model for KL.
        reward_fn: Reward function.
        optimizer: Optax optimizer.
        config: PPO configuration.
    
    Example:
        config = PPOConfig(
            clip_ratio=0.2,
            kl_coef=0.1,
            entropy_coef=0.01,
        )
        trainer = PPOTrainer(
            model=model,
            value_model=value_head,
            ref_model=ref_model,
            reward_fn=reward_fn,
            optimizer=optimizer,
            config=config,
        )
        trainer.train(dataloader)
    """
    
    def __init__(
        self,
        model: "LMModel",
        value_model: nnx.Module,
        ref_model: "LMModel",
        reward_fn: Callable[[Array, Array], Array],
        optimizer: optax.GradientTransformation,
        config: PPOConfig,
    ):
        from linearnexus.models import LMModel
        
        self.model = model
        self.value_model = value_model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.config = config
        
        # Initialize optimizer states
        graphdef, state = nnx.split(model)
        params = nnx.to_pure_dict(state)
        self.opt_state = optimizer.init(params)
        
        # Value optimizer (separate)
        graphdef_v, state_v = nnx.split(value_model)
        value_params = nnx.to_pure_dict(state_v)
        self.value_opt_state = optimizer.init(value_params)
        
        self.step = 0
        self.metrics_history = []
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_gae(
        self,
        rewards: Array,
        values: Array,
        dones: Array,
    ) -> Tuple[Array, Array]:
        """Compute GAE (Generalized Advantage Estimation).
        
        Args:
            rewards: Reward values [batch, seq].
            values: Value estimates [batch, seq].
            dones: Episode done flags [batch, seq].
            
        Returns:
            Tuple of (advantages, returns).
        """
        gamma = self.config.gamma
        lam = self.config.gae_lambda
        
        batch_size, seq_len = rewards.shape
        advantages = jnp.zeros_like(rewards)
        
        # Bootstrap from last value
        last_value = values[:, -1]
        last_gae = 0.0
        
        # Compute GAE backwards
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = last_value
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]
            last_gae = delta + gamma * lam * (1 - dones[:, t]) * last_gae
            advantages = advantages.at[:, t].set(last_gae)
        
        returns = advantages + values
        
        return advantages, returns
    
    def ppo_loss(
        self,
        params: Dict,
        batch: Dict[str, Array],
    ) -> Tuple[Array, Dict[str, Array]]:
        """Compute PPO policy loss.
        
        Args:
            params: Policy parameters.
            batch: Training batch with old_log_probs, advantages, etc.
            
        Returns:
            Tuple of (loss, metrics).
        """
        # Get current log probs
        graphdef, old_state = nnx.split(self.model)
        nnx.replace_by_pure_dict(old_state, params)
        model = nnx.merge(graphdef, old_state)
        
        logits, _ = model(batch["input_ids"])
        log_probs = compute_log_probs(logits, batch["actions"])
        
        # Compute ratio
        ratio = jnp.exp(log_probs - batch["old_log_probs"])
        
        # Clipped surrogate loss
        advantages = batch["advantages"]
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        
        # Entropy bonus
        probs = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
        entropy_loss = -self.config.entropy_coef * jnp.mean(entropy)
        
        # KL penalty
        kl = kl_divergence(log_probs, batch["ref_log_probs"])
        kl_loss = self.config.kl_coef * kl
        
        loss = policy_loss + entropy_loss + kl_loss
        
        metrics = {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy": jnp.mean(entropy),
            "kl": kl,
            "clip_frac": jnp.mean(jnp.abs(ratio - 1) > self.config.clip_ratio),
        }
        
        return loss, metrics
    
    def value_loss(
        self,
        value_params: Dict,
        batch: Dict[str, Array],
    ) -> Tuple[Array, Dict[str, Array]]:
        """Compute value function loss.
        
        Args:
            value_params: Value network parameters.
            batch: Training batch with returns.
            
        Returns:
            Tuple of (loss, metrics).
        """
        # Get value predictions
        graphdef, old_state = nnx.split(self.value_model)
        nnx.replace_by_pure_dict(old_state, value_params)
        value_model = nnx.merge(graphdef, old_state)
        
        values = value_model(batch["input_ids"])
        
        # Value clipping (optional)
        if "old_values" in batch:
            clipped_values = batch["old_values"] + jnp.clip(
                values - batch["old_values"],
                -self.config.value_clip,
                self.config.value_clip,
            )
            value_loss1 = (values - batch["returns"]) ** 2
            value_loss2 = (clipped_values - batch["returns"]) ** 2
            value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss1, value_loss2))
        else:
            value_loss = 0.5 * jnp.mean((values - batch["returns"]) ** 2)
        
        return value_loss, {"value_loss": value_loss}
    
    def train_step(
        self,
        batch: Dict[str, Array],
    ) -> Dict[str, Array]:
        """Single PPO training step.
        
        Args:
            batch: Training batch.
            
        Returns:
            Metrics dict.
        """
        # Get current params
        graphdef, state = nnx.split(self.model)
        params = nnx.to_pure_dict(state)
        
        graphdef_v, state_v = nnx.split(self.value_model)
        value_params = nnx.to_pure_dict(state_v)
        
        all_metrics = []
        
        # Multiple PPO epochs per batch
        for epoch in range(self.config.n_epochs):
            # Policy update
            grad_fn = jax.value_and_grad(self.ppo_loss, has_aux=True)
            (loss, metrics), grads = grad_fn(params, batch)
            
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # Value update
            grad_fn_v = jax.value_and_grad(self.value_loss, has_aux=True)
            (v_loss, v_metrics), v_grads = grad_fn_v(value_params, batch)
            
            v_updates, self.value_opt_state = self.optimizer.update(
                v_grads, self.value_opt_state, value_params
            )
            value_params = optax.apply_updates(value_params, v_updates)
            
            all_metrics.append({**metrics, **v_metrics})
        
        # Update models
        _, state = nnx.split(self.model)
        nnx.replace_by_pure_dict(state, params)
        self.model = nnx.merge(graphdef, state)
        
        _, state_v = nnx.split(self.value_model)
        nnx.replace_by_pure_dict(state_v, value_params)
        self.value_model = nnx.merge(graphdef_v, state_v)
        
        # Average metrics over epochs
        final_metrics = {}
        for key in all_metrics[0].keys():
            final_metrics[key] = sum(float(m[key]) for m in all_metrics) / len(all_metrics)
        
        return final_metrics
    
    def train(
        self,
        dataloader: Iterator[Dict[str, Array]],
        key: Optional[Array] = None,
    ) -> Dict[str, List[float]]:
        """Run PPO training loop.
        
        Args:
            dataloader: Iterator yielding training batches with:
                - input_ids: Token IDs
                - actions: Action tokens
                - rewards: Reward values
                - dones: Episode done flags
                - old_log_probs: Log probs from behavior policy
                - ref_log_probs: Log probs from reference model
            key: Random key (uses config seed if None).
            
        Returns:
            Dict of metric histories.
        """
        if key is None:
            key = jax.random.PRNGKey(self.config.seed)
        
        print(f"Starting PPO training for {self.config.max_steps} steps...")
        print(f"  Clip ratio: {self.config.clip_ratio}")
        print(f"  KL coef: {self.config.kl_coef}")
        print(f"  Entropy coef: {self.config.entropy_coef}")
        print(f"  PPO epochs: {self.config.n_epochs}")
        
        start_time = time.time()
        
        for step in range(self.step, self.config.max_steps):
            # Get batch
            try:
                batch = next(dataloader)
            except StopIteration:
                dataloader = iter(dataloader)
                batch = next(dataloader)
            
            # Training step
            metrics = self.train_step(batch)
            
            self.step = step + 1
            self.metrics_history.append({"step": self.step, **metrics})
            
            # Logging
            if self.step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Step {self.step}/{self.config.max_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f} | "
                    f"KL: {metrics['kl']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # Checkpointing
            if self.step % self.config.save_interval == 0:
                ckpt_manager = CheckpointManager(
                    self.output_dir,
                    max_to_keep=5,
                    best_metric="loss",
                    best_mode="min",
                )
                ckpt_manager.save(
                    step=self.step,
                    model=self.model,
                    metrics=metrics,
                )
        
        print(f"PPO training complete!")
        return {"metrics": self.metrics_history}
