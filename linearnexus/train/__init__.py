"""Training loops for LLM fine-tuning and reinforcement learning.

Implements multiple training paradigms:
- SFT (Supervised Fine-Tuning): Standard next-token prediction
- GRPO (Group Relative Policy Optimization): RL without value network
- PPO (Proximal Policy Optimization): Full RL with value head

All trainers use a unified interface and integrate with:
- Custom optimizers (AdamW, Muon, Sophia)
- Gradient accumulation and clipping
- Checkpointing and logging
- Mixed precision (via JAX dtype)

Example:
    from linearnexus.train import SFTTrainer, SFTConfig
    
    config = SFTConfig(
        learning_rate=3e-4,
        batch_size=4,
        max_steps=10000,
    )
    trainer = SFTTrainer(model, optimizer, config)
    trainer.train(dataloader)
"""

# Base classes and utilities
from linearnexus.train.base import (
    TrainConfig,
    cross_entropy_loss,
    compute_log_probs,
    kl_divergence,
)

# SFT
from linearnexus.train.sft import SFTTrainer, SFTConfig

# GRPO
from linearnexus.train.grpo import GRPOTrainer, GRPOConfig

# PPO
from linearnexus.train.ppo import PPOTrainer, PPOConfig

# Factory
from linearnexus.train.factory import create_trainer

__all__ = [
    # Base
    "TrainConfig",
    "cross_entropy_loss",
    "compute_log_probs",
    "kl_divergence",
    # SFT
    "SFTTrainer",
    "SFTConfig",
    # GRPO
    "GRPOTrainer",
    "GRPOConfig",
    # PPO
    "PPOTrainer",
    "PPOConfig",
    # Factory
    "create_trainer",
]
