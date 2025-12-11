"""Supervised Fine-Tuning (SFT) Trainer.

Standard next-token prediction training with:
- Cross-entropy loss
- Gradient accumulation
- Learning rate warmup + cosine decay
- Checkpointing
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from linearnexus.train.base import (
    SFTConfig,
    cross_entropy_loss,
)
from linearnexus.checkpoint import CheckpointManager

Array = jax.Array


class SFTTrainer:
    """Supervised Fine-Tuning trainer.
    
    Standard next-token prediction training with:
    - Cross-entropy loss
    - Gradient accumulation
    - Learning rate warmup + cosine decay
    - Checkpointing
    
    Args:
        model: Language model to train.
        optimizer: Optax optimizer.
        config: Training configuration.
    
    Example:
        config = SFTConfig(
            learning_rate=3e-4,
            batch_size=4,
            max_steps=10000,
        )
        trainer = SFTTrainer(model, optimizer, config)
        trainer.train(dataloader)
    """
    
    def __init__(
        self,
        model: "LMModel",
        optimizer: optax.GradientTransformation,
        config: SFTConfig,
    ):
        from linearnexus.models import LMModel
        
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        # Initialize optimizer state
        graphdef, state = nnx.split(model)
        params = nnx.to_pure_dict(state)
        self.opt_state = optimizer.init(params)
        
        self.step = 0
        self.metrics_history = []
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def loss_fn(
        self,
        params: Dict,
        batch: Dict[str, Array],
    ) -> Tuple[Array, Dict[str, Array]]:
        """Compute loss for a batch.
        
        Args:
            params: Model parameters (pure dict).
            batch: Dict with "input_ids" and "labels".
            
        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Reconstruct model with params
        graphdef, old_state = nnx.split(self.model)
        nnx.replace_by_pure_dict(old_state, params)
        model = nnx.merge(graphdef, old_state)
        
        # Forward pass
        logits, _ = model(batch["input_ids"])
        
        # Compute loss
        loss = cross_entropy_loss(logits, batch["labels"])
        
        # Compute perplexity
        ppl = jnp.exp(loss)
        
        return loss, {"loss": loss, "perplexity": ppl}
    
    @jax.jit
    def train_step(
        self,
        params: Dict,
        opt_state: optax.OptState,
        batch: Dict[str, Array],
    ) -> Tuple[Dict, optax.OptState, Dict[str, Array]]:
        """Single training step.
        
        Args:
            params: Model parameters.
            opt_state: Optimizer state.
            batch: Training batch.
            
        Returns:
            Tuple of (new_params, new_opt_state, metrics).
        """
        # Compute gradients
        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(params, batch)
        
        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Add gradient norm to metrics
        grad_norm = optax.global_norm(grads)
        metrics["grad_norm"] = grad_norm
        
        return new_params, new_opt_state, metrics
    
    def train(
        self,
        dataloader: Iterator[Dict[str, Array]],
        eval_dataloader: Optional[Iterator[Dict[str, Array]]] = None,
    ) -> Dict[str, List[float]]:
        """Run training loop.
        
        Args:
            dataloader: Training data iterator.
            eval_dataloader: Optional evaluation data iterator.
            
        Returns:
            Dict of metric histories.
        """
        print(f"Starting SFT training for {self.config.max_steps} steps...")
        
        # Get initial params
        graphdef, state = nnx.split(self.model)
        params = nnx.to_pure_dict(state)
        
        start_time = time.time()
        accumulated_loss = 0.0
        accumulated_steps = 0
        
        for step in range(self.step, self.config.max_steps):
            # Get batch
            try:
                batch = next(dataloader)
            except StopIteration:
                # Reset dataloader (epoch boundary)
                dataloader = iter(dataloader)
                batch = next(dataloader)
            
            # Training step
            params, self.opt_state, metrics = self.train_step(
                params, self.opt_state, batch
            )
            
            accumulated_loss += float(metrics["loss"])
            accumulated_steps += 1
            self.step = step + 1
            
            # Logging
            if self.step % self.config.log_interval == 0:
                avg_loss = accumulated_loss / accumulated_steps
                elapsed = time.time() - start_time
                steps_per_sec = self.step / elapsed
                
                print(
                    f"Step {self.step}/{self.config.max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"PPL: {jnp.exp(avg_loss):.2f} | "
                    f"Steps/sec: {steps_per_sec:.2f}"
                )
                
                self.metrics_history.append({
                    "step": self.step,
                    "loss": avg_loss,
                    "perplexity": float(jnp.exp(avg_loss)),
                    "grad_norm": float(metrics["grad_norm"]),
                })
                
                accumulated_loss = 0.0
                accumulated_steps = 0
            
            # Evaluation
            if eval_dataloader and self.step % self.config.eval_interval == 0:
                eval_loss = self.evaluate(params, eval_dataloader)
                print(f"Eval Loss: {eval_loss:.4f} | Eval PPL: {jnp.exp(eval_loss):.2f}")
            
            # Checkpointing
            if self.step % self.config.save_interval == 0:
                # Update model with new params
                _, state = nnx.split(self.model)
                nnx.replace_by_pure_dict(state, params)
                self.model = nnx.merge(graphdef, state)
                
                # Use CheckpointManager for saving
                ckpt_manager = CheckpointManager(
                    self.output_dir,
                    max_to_keep=5,
                    best_metric="loss",
                    best_mode="min",
                )
                ckpt_manager.save(
                    step=self.step,
                    model=self.model,
                    metrics={"loss": float(metrics["loss"])},
                )
        
        # Final update
        _, state = nnx.split(self.model)
        nnx.replace_by_pure_dict(state, params)
        self.model = nnx.merge(graphdef, state)
        
        print(f"Training complete! Final loss: {metrics['loss']:.4f}")
        return {"metrics": self.metrics_history}
    
    def evaluate(
        self,
        params: Dict,
        dataloader: Iterator[Dict[str, Array]],
        num_batches: int = 10,
    ) -> float:
        """Evaluate model on validation data.
        
        Args:
            params: Model parameters.
            dataloader: Evaluation data iterator.
            num_batches: Number of batches to evaluate.
            
        Returns:
            Average loss.
        """
        total_loss = 0.0
        
        for i in range(num_batches):
            try:
                batch = next(dataloader)
            except StopIteration:
                break
            
            loss, _ = self.loss_fn(params, batch)
            total_loss += float(loss)
        
        return total_loss / max(i + 1, 1)
