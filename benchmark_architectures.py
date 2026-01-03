#!/usr/bin/env python3
"""Benchmark all LinearNexus architectures under identical conditions.

Trains all 8 architectures (GPT, Mamba, Mamba2, DeltaNet, Gated DeltaNet,
RWKV6, RWKV7, KDA) with standardized model sizes for fair comparison.

Usage:
    python benchmark_architectures.py
    python benchmark_architectures.py --max-steps 1000
    python benchmark_architectures.py --arch gpt mamba rwkv7
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

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
# Standardized Model Configurations (~3-5M params each for fair comparison)
# =============================================================================

def get_standardized_configs(vocab_size: int = 65) -> Dict[str, ModelConfig]:
    """Get model configs with standardized sizes for fair comparison."""
    
    hidden_size = 256
    n_layers = 6
    
    return {
        "gpt": ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=8,
            head_dim=32,
            block_pattern=["attention"],
            intermediate_size=hidden_size * 4,
        ),
        "mamba": ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            block_pattern=["mamba"],
            state_size=16,
            conv_kernel=4,
            intermediate_size=hidden_size * 2,
        ),
        "mamba2": ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            block_pattern=["mamba2"],
            mamba2_heads=8,
            mamba2_head_dim=32,
            mamba2_state_size=64,
            mamba2_n_groups=1,
        ),
        "deltanet": ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=8,
            block_pattern=["deltanet"],
            deltanet_heads=8,
            deltanet_expand_k=1.0,
            deltanet_expand_v=1.0,
            deltanet_use_beta=True,
            deltanet_use_gate=False,
            deltanet_use_short_conv=True,
            deltanet_chunk_size=32,
        ),
        "gated_deltanet": ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            block_pattern=["gated_deltanet"],
            gated_deltanet_heads=4,
            gated_deltanet_v_heads=4,
            gated_deltanet_head_dim=64,
            gated_deltanet_expand_v=2.0,
            gated_deltanet_use_short_conv=True,
            gated_deltanet_use_gate=True,
        ),
        "rwkv6": ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=8,
            block_pattern=["rwkv6"],
            rwkv6_heads=8,
            rwkv6_proj_low_rank_dim=32,
            rwkv6_gate_low_rank_dim=64,
            rwkv6_intermediate_size=hidden_size * 4,
        ),
        "rwkv7": ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=8,
            block_pattern=["rwkv7"],
            rwkv7_heads=8,
            rwkv7_head_dim=32,
        ),
        "kda": ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=8,
            block_pattern=["kda"],
            kda_heads=4,
            kda_v_heads=4,
            kda_head_dim=64,
            kda_expand_v=1.0,
            kda_use_short_conv=True,
        ),
    }


@dataclass
class BenchmarkResult:
    """Result from benchmarking an architecture."""
    arch_name: str
    param_count: int
    final_loss: float
    final_perplexity: float
    loss_history: List[float]
    wall_time: float
    tokens_per_sec: float
    passed: bool
    error: Optional[str] = None


def benchmark_architecture(
    arch_name: str,
    config: ModelConfig,
    dataloader: DataLoader,
    max_steps: int = 500,
    lr: float = 3e-4,
    warmup_steps: int = 50,
    log_interval: int = 25,
    batch_size: int = 8,
    seq_len: int = 128,
) -> BenchmarkResult:
    """Benchmark a single architecture."""
    
    print(f"\n{'='*60}")
    print(f"Architecture: {arch_name.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    loss_history = []
    total_tokens = 0
    
    try:
        # Create model
        rngs = nnx.Rngs(42)
        model = LMModel(config, rngs=rngs)
        param_count = model.count_params()
        print(f"  Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
        
        # Create optimizer
        optimizer = create_optimizer(
            "adamw",
            learning_rate=lr,
            total_steps=max_steps,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            grad_clip=1.0,
        )
        
        # Split model into params and rest (rngs, etc.)
        graphdef, params_state, rest_state = nnx.split(model, nnx.Param, ...)
        params = nnx.to_pure_dict(params_state)
        opt_state = optimizer.init(params)
        
        # Define loss function
        def loss_fn(params, batch, graphdef, params_state, rest_state):
            nnx.replace_by_pure_dict(params_state, params)
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
        train_start = time.time()
        
        for step in range(max_steps):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Training step
            params, opt_state, loss = train_step(params, opt_state, batch, graphdef, params_state, rest_state)
            total_tokens += batch_size * seq_len
            
            # Log
            if step == 0 or (step + 1) % log_interval == 0 or step == max_steps - 1:
                loss_val = float(loss)
                ppl = float(jnp.exp(loss))
                loss_history.append(loss_val)
                elapsed = time.time() - train_start
                tokens_sec = total_tokens / elapsed if elapsed > 0 else 0
                print(f"  Step {step + 1:4d}: loss={loss_val:.4f}, ppl={ppl:.2f}, tokens/s={tokens_sec:.0f}")
                
                # Check for NaN/Inf
                if not jnp.isfinite(loss):
                    return BenchmarkResult(
                        arch_name=arch_name,
                        param_count=param_count,
                        final_loss=float('nan'),
                        final_perplexity=float('nan'),
                        loss_history=loss_history,
                        wall_time=time.time() - start_time,
                        tokens_per_sec=0,
                        passed=False,
                        error="Loss became NaN/Inf",
                    )
        
        wall_time = time.time() - start_time
        train_time = time.time() - train_start
        tokens_per_sec = total_tokens / train_time if train_time > 0 else 0
        
        final_loss = loss_history[-1]
        final_ppl = float(jnp.exp(final_loss))
        
        print(f"  Final: loss={final_loss:.4f}, ppl={final_ppl:.2f}")
        print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
        print(f"  Wall time: {wall_time:.1f}s")
        
        return BenchmarkResult(
            arch_name=arch_name,
            param_count=param_count,
            final_loss=final_loss,
            final_perplexity=final_ppl,
            loss_history=loss_history,
            wall_time=wall_time,
            tokens_per_sec=tokens_per_sec,
            passed=True,
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return BenchmarkResult(
            arch_name=arch_name,
            param_count=0,
            final_loss=float('nan'),
            final_perplexity=float('nan'),
            loss_history=loss_history,
            wall_time=time.time() - start_time,
            tokens_per_sec=0,
            passed=False,
            error=str(e),
        )


def print_results(results: List[BenchmarkResult], batch_size: int, seq_len: int, max_steps: int):
    """Print ranked comparison table."""
    
    # Sort by final loss (best first)
    valid_results = [r for r in results if r.passed and not jnp.isnan(r.final_loss)]
    failed_results = [r for r in results if not r.passed or jnp.isnan(r.final_loss)]
    sorted_results = sorted(valid_results, key=lambda x: x.final_loss)
    
    print("\n" + "=" * 100)
    print(" " * 30 + "ARCHITECTURE BENCHMARK RESULTS")
    print("=" * 100)
    print(f"Training: {max_steps} steps, batch_size={batch_size}, seq_len={seq_len}")
    print("=" * 100)
    
    # Header
    print(f"{'Rank':<6}{'Architecture':<18}{'Params':<10}{'Final Loss':<12}{'Perplexity':<12}{'Time':<10}{'Tokens/s':<12}")
    print("-" * 100)
    
    # Valid results
    for rank, result in enumerate(sorted_results, 1):
        param_str = f"{result.param_count / 1e6:.2f}M"
        time_str = f"{result.wall_time:.1f}s"
        tokens_str = f"{result.tokens_per_sec:,.0f}"
        print(
            f"{rank:<6}"
            f"{result.arch_name:<18}"
            f"{param_str:<10}"
            f"{result.final_loss:<12.4f}"
            f"{result.final_perplexity:<12.2f}"
            f"{time_str:<10}"
            f"{tokens_str:<12}"
        )
    
    # Failed results
    for result in failed_results:
        print(f"{'--':<6}{result.arch_name:<18}{'FAILED':<10}{'--':<12}{'--':<12}{'--':<10}{'--':<12}")
        if result.error:
            print(f"       └─ Error: {result.error[:60]}...")
    
    print("=" * 100)
    
    # Best architecture
    if sorted_results:
        best = sorted_results[0]
        print(f"\n🏆 Best Architecture: {best.arch_name.upper()} (loss={best.final_loss:.4f}, ppl={best.final_perplexity:.2f})")
    
    return sorted_results


def save_results(results: List[BenchmarkResult], output_path: Path):
    """Save results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(jax.devices()[0]),
        "results": [asdict(r) for r in results]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all architectures")
    parser.add_argument(
        "--arch", nargs="+", default=None,
        choices=["gpt", "mamba", "mamba2", "deltanet", "gated_deltanet", "rwkv6", "rwkv7", "kda"],
        help="Specific architectures to benchmark (default: all)",
    )
    parser.add_argument("--max-steps", type=int, default=500, help="Training steps per architecture")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LinearNexus Architecture Benchmark")
    print("=" * 60)
    
    # Check device
    devices = jax.devices()
    device = devices[0]
    print(f"\nDevice: {device}")
    if device.device_kind == "cpu":
        print("⚠ WARNING: Running on CPU!")
    else:
        print(f"✓ Running on: {device.device_kind}")
    
    # Download data
    print("\nPreparing Shakespeare dataset...")
    data_path = download_shakespeare()
    
    # Create tokenizer
    tokenizer = CharTokenizer.from_file(data_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset and dataloader
    print(f"Creating dataset: seq_len={args.seq_len}, batch_size={args.batch_size}")
    dataset = TextDataset(data_path, tokenizer, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Get standardized configs
    configs = get_standardized_configs(tokenizer.vocab_size)
    
    # Determine architectures to benchmark
    archs_to_test = args.arch if args.arch else list(configs.keys())
    print(f"\nBenchmarking: {archs_to_test}")
    print(f"Training: {args.max_steps} steps, lr={args.lr}, warmup={args.warmup_steps}")
    
    # Run benchmarks
    results = []
    for arch_name in archs_to_test:
        result = benchmark_architecture(
            arch_name=arch_name,
            config=configs[arch_name],
            dataloader=dataloader,
            max_steps=args.max_steps,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        results.append(result)
    
    # Print and save results
    print_results(results, args.batch_size, args.seq_len, args.max_steps)
    save_results(results, Path(args.output))
    
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
