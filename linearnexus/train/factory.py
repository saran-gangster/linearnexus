"""Factory function for creating trainers.

Provides a unified interface to create any trainer type.
"""

from __future__ import annotations

from typing import Union

import optax

from linearnexus.train.base import TrainConfig
from linearnexus.train.sft import SFTTrainer, SFTConfig
from linearnexus.train.grpo import GRPOTrainer, GRPOConfig
from linearnexus.train.ppo import PPOTrainer, PPOConfig


def create_trainer(
    trainer_type: str,
    model: "LMModel",
    optimizer: optax.GradientTransformation,
    config: TrainConfig,
    **kwargs,
) -> Union[SFTTrainer, GRPOTrainer, PPOTrainer]:
    """Factory function for trainers.
    
    Args:
        trainer_type: "sft", "grpo", or "ppo".
        model: Model to train.
        optimizer: Optax optimizer.
        config: Training configuration (should match trainer_type).
        **kwargs: Additional trainer-specific arguments.
            For GRPO: ref_model (optional), reward_fns (list of functions)
            For PPO: value_model, ref_model, reward_fn
        
    Returns:
        Trainer instance.
    
    Examples:
        # SFT
        trainer = create_trainer("sft", model, optimizer, SFTConfig())
        
        # GRPO
        trainer = create_trainer(
            "grpo", 
            model, 
            optimizer, 
            GRPOConfig(),
            reward_fns=[my_reward_fn],
        )
        
        # PPO
        trainer = create_trainer(
            "ppo",
            model,
            optimizer,
            PPOConfig(),
            value_model=value_head,
            ref_model=ref_model,
            reward_fn=reward_fn,
        )
    
    Raises:
        ValueError: If trainer_type is unknown.
        KeyError: If required kwargs are missing for GRPO/PPO.
    """
    from linearnexus.models import LMModel
    
    if trainer_type == "sft":
        return SFTTrainer(model, optimizer, config)
    elif trainer_type == "grpo":
        return GRPOTrainer(
            model,
            kwargs.get("ref_model"),  # Optional, can be None
            kwargs["reward_fns"],      # List of reward functions
            optimizer,
            config,
        )
    elif trainer_type == "ppo":
        return PPOTrainer(
            model,
            kwargs["value_model"],
            kwargs["ref_model"],
            kwargs["reward_fn"],
            optimizer,
            config,
        )
    else:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Supported types: 'sft', 'grpo', 'ppo'"
        )
