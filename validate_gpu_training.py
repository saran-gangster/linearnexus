#!/usr/bin/env python3
"""Validate GPU training convergence for all LinearNexus architectures.

Tests that each architecture (GPT, Mamba, Mamba2, DeltaNet, Gated DeltaNet,
RWKV6, RWKV7, KDA) can train successfully on GPU with loss decreasing.

Usage:
    python validate_gpu_training.py
    python validate_gpu_training.py --arch gpt mamba  # Test specific architectures
    python validate_gpu_training.py --max-steps 200   # More training steps
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from linearnexus.models import ModelConfig, LMModel
from linearnexus.data import CharTokenizer, TextDataset, DataLoader, download_shakespeare
from linearnexus.optim import create_optimizer
from linearnexus.train.base import cross_entropy_loss


# =============================================================================
# Tiny Model Configurations (~500K-2M params for fast GPU testing)
# =============================================================================

TINY_CONFIGS = {
    "gpt": ModelConfig(
        vocab_size=65,  # Will be updated from tokenizer
        hidden_size=128,
        n_layers=4,
        n_heads=4,
        head_dim=32,
        block_pattern=["attention"],
        intermediate_size=512,
    ),
    "mamba": ModelConfig(
        vocab_size=65,
        hidden_size=128,
        n_layers=4,
        block_pattern=["mamba"],
        state_size=8,
        conv_kernel=4,
        intermediate_size=256,
    ),
    "mamba2": ModelConfig(
        vocab_size=65,
        hidden_size=128,
        n_layers=4,
        block_pattern=["mamba2"],
        mamba2_heads=4,
        mamba2_head_dim=32,
        mamba2_state_size=64,
        mamba2_n_groups=1,
    ),
    "deltanet": ModelConfig(
        vocab_size=65,
        hidden_size=128,
        n_layers=4,
        n_heads=4,
        block_pattern=["deltanet"],
        deltanet_heads=4,
        deltanet_expand_k=1.0,
        deltanet_expand_v=1.0,
        deltanet_use_beta=True,
        deltanet_use_gate=False,
        deltanet_use_short_conv=True,
        deltanet_chunk_size=32,
    ),
    "gated_deltanet": ModelConfig(
        vocab_size=65,
        hidden_size=128,
        n_layers=4,
        block_pattern=["gated_deltanet"],
        gated_deltanet_heads=2,
        gated_deltanet_v_heads=2,
        gated_deltanet_head_dim=64,
        gated_deltanet_expand_v=2.0,
        gated_deltanet_use_short_conv=True,
        gated_deltanet_use_gate=True,
    ),
    "rwkv6": ModelConfig(
        vocab_size=65,
        hidden_size=128,
        n_layers=4,
        n_heads=4,
        block_pattern=["rwkv6"],
        rwkv6_heads=4,
        rwkv6_proj_low_rank_dim=16,
        rwkv6_gate_low_rank_dim=32,
        rwkv6_intermediate_size=512,
    ),
    "rwkv7": ModelConfig(
        vocab_size=65,
        hidden_size=128,
        n_layers=4,
        n_heads=4,
        block_pattern=["rwkv7"],
        rwkv7_heads=4,
        rwkv7_head_dim=32,
    ),
    "kda": ModelConfig(
        vocab_size=65,
        hidden_size=128,
        n_layers=4,
        n_heads=4,
        block_pattern=["kda"],
        kda_heads=1,
        kda_v_heads=1,
        kda_head_dim=128,
        kda_expand_v=1.0,
        kda_use_short_conv=True,
    ),
}


@dataclass
class TrainingResult:
    """Result from training an architecture."""
    arch_name: str
    param_count: int
    losses: List[float]
    wall_time: float
    passed: bool
    error: Optional[str] = None


def train_architecture(
    arch_name: str,
    config: ModelConfig,
    dataloader: DataLoader,
    max_steps: int = 100,
    lr: float = 1e-3,
    log_interval: int = 25,
) -> TrainingResult:
    """Train a single architecture and return results.
    
    Args:
        arch_name: Name of the architecture.
        config: Model configuration.
        dataloader: Data loader with training data.
        max_steps: Number of training steps.
        lr: Learning rate.
        log_interval: How often to log losses.
    
    Returns:
        TrainingResult with loss history and pass/fail status.
    """
    print(f"\n{'='*60}")
    print(f"Architecture: {arch_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    losses = []
    
    try:
        # Create model
        rngs = nnx.Rngs(42)
        model = LMModel(config, rngs=rngs)
        param_count = model.count_params()
        print(f"  Model size: {param_count:,} params ({param_count/1e6:.2f}M)")
        
        # Create optimizer
        optimizer = create_optimizer(
            "adamw",
            learning_rate=lr,
            total_steps=max_steps,
            warmup_steps=10,
            weight_decay=0.01,
            grad_clip=1.0,
        )
        
        # Split model into graphdef, trainable params, and rest (rngs, etc.)
        # This separates float params from uint32 rngs
        graphdef, params_state, rest_state = nnx.split(model, nnx.Param, ...)
        params = nnx.to_pure_dict(params_state)
        opt_state = optimizer.init(params)
        
        # Define loss function with only float params as first arg
        def loss_fn(params, batch, graphdef, params_state, rest_state):
            # Update params_state with new values
            nnx.replace_by_pure_dict(params_state, params)
            # Merge all state back
            merged_model = nnx.merge(graphdef, params_state, rest_state)
            logits, _ = merged_model(batch["input_ids"])
            return cross_entropy_loss(logits, batch["labels"])
        
        # JIT compile training step
        @jax.jit
        def train_step(params, opt_state, batch, graphdef, params_state, rest_state):
            loss, grads = jax.value_and_grad(loss_fn)(params, batch, graphdef, params_state, rest_state)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        # Training loop
        data_iter = iter(dataloader)
        
        for step in range(max_steps):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Training step
            params, opt_state, loss = train_step(params, opt_state, batch, graphdef, params_state, rest_state)
            
            # Log
            if step == 0 or (step + 1) % log_interval == 0 or step == max_steps - 1:
                loss_val = float(loss)
                losses.append(loss_val)
                print(f"  Step {step + 1:3d}: loss={loss_val:.4f}")
                
                # Check for NaN/Inf
                if not jnp.isfinite(loss):
                    return TrainingResult(
                        arch_name=arch_name,
                        param_count=param_count,
                        losses=losses,
                        wall_time=time.time() - start_time,
                        passed=False,
                        error="Loss became NaN/Inf",
                    )
        
        wall_time = time.time() - start_time
        
        # Check convergence: loss should decrease by at least 20%
        init_loss = losses[0]
        final_loss = losses[-1]
        reduction = (init_loss - final_loss) / init_loss * 100
        passed = reduction >= 20.0
        
        print(f"  Loss reduction: {reduction:.1f}% {'✓' if passed else '✗'}")
        print(f"  Wall time: {wall_time:.1f}s")
        
        return TrainingResult(
            arch_name=arch_name,
            param_count=param_count,
            losses=losses,
            wall_time=wall_time,
            passed=passed,
            error=None if passed else f"Only {reduction:.1f}% loss reduction (need ≥20%)",
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return TrainingResult(
            arch_name=arch_name,
            param_count=0,
            losses=losses,
            wall_time=time.time() - start_time,
            passed=False,
            error=str(e),
        )


def print_summary(results: List[TrainingResult]) -> bool:
    """Print summary table and return overall pass/fail."""
    print("\n" + "=" * 80)
    print(" " * 25 + "RESULTS SUMMARY")
    print("=" * 80)
    
    # Header
    print(f"{'Architecture':<18} {'Params':<8} {'Init Loss':<10} {'Final Loss':<11} {'Reduction':<10} {'Status':<8}")
    print("-" * 80)
    
    all_passed = True
    for result in results:
        if result.losses:
            init_loss = result.losses[0]
            final_loss = result.losses[-1]
            reduction = (init_loss - final_loss) / init_loss * 100
        else:
            init_loss = 0.0
            final_loss = 0.0
            reduction = 0.0
        
        status = "PASS" if result.passed else "FAIL"
        status_symbol = "✓" if result.passed else "✗"
        
        if not result.passed:
            all_passed = False
        
        param_str = f"{result.param_count / 1000:.0f}K" if result.param_count > 0 else "N/A"
        
        print(
            f"{result.arch_name:<18} "
            f"{param_str:<8} "
            f"{init_loss:<10.4f} "
            f"{final_loss:<11.4f} "
            f"{reduction:<10.1f}% "
            f"{status} {status_symbol}"
        )
        
        if result.error and not result.passed:
            print(f"   └─ Error: {result.error}")
    
    print("=" * 80)
    
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    
    if all_passed:
        print(f"\n✓ All {total_count}/{total_count} architectures PASSED!")
    else:
        print(f"\n✗ {passed_count}/{total_count} architectures passed, {total_count - passed_count} failed")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate GPU training for all architectures")
    parser.add_argument(
        "--arch", nargs="+", default=None,
        choices=list(TINY_CONFIGS.keys()),
        help="Specific architectures to test (default: all)",
    )
    parser.add_argument("--max-steps", type=int, default=100, help="Training steps per architecture")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LinearNexus GPU Training Validation")
    print("=" * 60)
    
    # Check device
    devices = jax.devices()
    print(f"\nDevices: {devices}")
    if devices[0].device_kind == "cpu":
        print("⚠ WARNING: Running on CPU, not GPU!")
    else:
        print(f"✓ Running on: {devices[0].device_kind}")
    
    # Download Shakespeare data
    print("\nDownloading Shakespeare dataset...")
    data_path = download_shakespeare()
    
    # Create tokenizer
    tokenizer = CharTokenizer.from_file(data_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset and dataloader
    print(f"Creating dataset with seq_len={args.seq_len}...")
    dataset = TextDataset(data_path, tokenizer, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Determine which architectures to test
    archs_to_test = args.arch if args.arch else list(TINY_CONFIGS.keys())
    print(f"\nTesting architectures: {archs_to_test}")
    
    # Run training for each architecture
    results = []
    for arch_name in archs_to_test:
        config = TINY_CONFIGS[arch_name]
        # Update vocab size from tokenizer
        config = ModelConfig(**{**config.to_dict(), "vocab_size": tokenizer.vocab_size})
        
        result = train_architecture(
            arch_name=arch_name,
            config=config,
            dataloader=dataloader,
            max_steps=args.max_steps,
            lr=args.lr,
        )
        results.append(result)
    
    # Print summary
    all_passed = print_summary(results)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
