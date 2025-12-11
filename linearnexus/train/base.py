"""Base training utilities, configs, and loss functions.

This module provides:
- Base configuration classes (TrainConfig and algorithm-specific configs)
- Core loss functions (cross_entropy, log_probs, KL divergence)
- Checkpointing utilities (save/load)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from linearnexus.core import ConfigBase

Array = jax.Array


# -----------------------------------------------------------------------------
# Training Configurations
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig(ConfigBase):
    """Base training configuration."""
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    
    # Batching
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    
    # Training loop
    max_steps: int = 10000
    eval_interval: int = 500
    log_interval: int = 10
    save_interval: int = 1000
    
    # Checkpointing
    output_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    
    # Misc
    seed: int = 42
    dtype: str = "float32"


@dataclass
class SFTConfig(TrainConfig):
    """Supervised Fine-Tuning configuration."""
    pass


@dataclass
class GRPOConfig(TrainConfig):
    """GRPO (Group Relative Policy Optimization) configuration.
    
    GRPO is a simplified RL algorithm that doesn't require a value network.
    Instead, it uses relative rewards within a group of samples.
    
    Reference: https://github.com/joey00072/nanoGRPO
    """
    
    # GRPO specific
    group_size: int = 8          # Number of samples per prompt for reward comparison
    micro_group_size: int = 2    # Micro-batching for memory efficiency
    beta: float = 0.0            # KL divergence penalty coefficient (0 = no KL penalty)
    epsilon: float = 0.1         # PPO-style clipping ratio
    max_gen_tokens: int = 512    # Maximum tokens to generate per completion
    temperature: float = 0.9     # Sampling temperature for generation


@dataclass 
class PPOConfig(TrainConfig):
    """PPO (Proximal Policy Optimization) configuration.
    
    Full PPO with value network, GAE, and clipping.
    """
    
    # PPO specific
    clip_ratio: float = 0.2      # Policy clipping ratio
    value_clip: float = 0.2      # Value function clipping
    kl_coef: float = 0.1         # KL penalty coefficient
    entropy_coef: float = 0.01   # Entropy bonus coefficient
    gae_lambda: float = 0.95     # GAE lambda
    gamma: float = 0.99          # Discount factor
    n_epochs: int = 4            # PPO epochs per batch
    value_loss_coef: float = 0.5 # Value loss weight


# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------

def cross_entropy_loss(
    logits: Array,
    labels: Array,
    mask: Optional[Array] = None,
) -> Array:
    """Cross-entropy loss for language modeling.
    
    Args:
        logits: Model outputs [batch, seq, vocab].
        labels: Target token IDs [batch, seq].
        mask: Optional mask [batch, seq] (1 = include, 0 = ignore).
        
    Returns:
        Scalar loss value.
    """
    vocab_size = logits.shape[-1]
    
    # Flatten for cross-entropy
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    
    # Compute per-token loss
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    token_loss = -jnp.take_along_axis(
        log_probs,
        labels_flat[:, None],
        axis=-1,
    ).squeeze(-1)
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.reshape(-1)
        token_loss = token_loss * mask_flat
        return jnp.sum(token_loss) / (jnp.sum(mask_flat) + 1e-8)
    
    return jnp.mean(token_loss)


def compute_log_probs(
    logits: Array,
    labels: Array,
) -> Array:
    """Compute per-token log probabilities.
    
    Args:
        logits: Model outputs [batch, seq, vocab].
        labels: Token IDs [batch, seq].
        
    Returns:
        Log probabilities [batch, seq].
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)


def kl_divergence(
    log_probs_p: Array,
    log_probs_q: Array,
    mask: Optional[Array] = None,
) -> Array:
    """KL divergence between two distributions.
    
    KL(P || Q) = sum(P * (log(P) - log(Q)))
    
    Args:
        log_probs_p: Log probs of P [batch, seq].
        log_probs_q: Log probs of Q [batch, seq].
        mask: Optional mask.
        
    Returns:
        Scalar KL divergence.
    """
    kl = jnp.exp(log_probs_p) * (log_probs_p - log_probs_q)
    
    if mask is not None:
        kl = kl * mask
        return jnp.sum(kl) / (jnp.sum(mask) + 1e-8)
    
    return jnp.mean(kl)
