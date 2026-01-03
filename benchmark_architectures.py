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
# Parameter-Matched Model Configurations
# =============================================================================


def _count_params_for_config(config: ModelConfig) -> int:
    """Instantiate an `LMModel` and return its parameter count.

    This is used for benchmarking config auto-tuning (no training).
    """
    model = LMModel(config, rngs=nnx.Rngs(0))
    return int(model.count_params())


def _arch_config(
    arch: str,
    *,
    vocab_size: int,
    hidden_size: int,
    n_layers: int,
) -> ModelConfig:
    """Create a ModelConfig for a given architecture at a given hidden size.

    Notes:
    - We choose head dims that keep shapes valid for each architecture.
    - We scale low-rank dims for RWKV6 with hidden_size to keep ratios stable.
    """
    if arch == "gpt":
        head_dim = 32
        n_heads = max(1, hidden_size // head_dim)
        hidden_size = n_heads * head_dim
        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            block_pattern=["attention"],
            intermediate_size=hidden_size * 4,
        )

    if arch == "mamba":
        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            block_pattern=["mamba"],
            state_size=16,
            conv_kernel=4,
            intermediate_size=hidden_size * 2,
        )

    if arch == "mamba2":
        head_dim = 32
        n_heads = max(1, hidden_size // head_dim)
        hidden_size = n_heads * head_dim
        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            block_pattern=["mamba2"],
            mamba2_heads=n_heads,
            mamba2_head_dim=head_dim,
            mamba2_state_size=64,
            mamba2_n_groups=1,
        )

    if arch == "deltanet":
        head_dim = 32
        n_heads = max(1, hidden_size // head_dim)
        hidden_size = n_heads * head_dim
        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            block_pattern=["deltanet"],
            deltanet_heads=n_heads,
            deltanet_expand_k=1.0,
            deltanet_expand_v=1.0,
            deltanet_use_beta=True,
            deltanet_use_gate=False,
            deltanet_use_short_conv=True,
            deltanet_chunk_size=32,
        )

    if arch == "gated_deltanet":
        # Keep Mamba2-style head dim (but smaller than the default preset)
        head_dim = 64
        n_heads = max(1, hidden_size // head_dim)
        hidden_size = n_heads * head_dim
        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            block_pattern=["gated_deltanet"],
            gated_deltanet_heads=n_heads,
            gated_deltanet_v_heads=n_heads,
            gated_deltanet_head_dim=head_dim,
            gated_deltanet_expand_v=2.0,
            gated_deltanet_use_short_conv=True,
            gated_deltanet_use_gate=True,
        )

    if arch == "rwkv6":
        head_dim = 32
        n_heads = max(1, hidden_size // head_dim)
        hidden_size = n_heads * head_dim
        proj_rank = max(8, hidden_size // 8)   # 256 -> 32
        gate_rank = max(8, hidden_size // 4)   # 256 -> 64
        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            block_pattern=["rwkv6"],
            rwkv6_heads=n_heads,
            rwkv6_proj_low_rank_dim=proj_rank,
            rwkv6_gate_low_rank_dim=gate_rank,
            rwkv6_intermediate_size=hidden_size * 4,
        )

    if arch == "rwkv7":
        head_dim = 32
        n_heads = max(1, hidden_size // head_dim)
        hidden_size = n_heads * head_dim
        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            block_pattern=["rwkv7"],
            rwkv7_heads=n_heads,
            rwkv7_head_dim=head_dim,
        )

    if arch == "kda":
        head_dim = 64
        n_heads = max(1, hidden_size // head_dim)
        hidden_size = n_heads * head_dim
        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=max(1, hidden_size // n_heads),
            block_pattern=["kda"],
            kda_heads=n_heads,
            kda_v_heads=n_heads,
            kda_head_dim=head_dim,
            kda_expand_v=1.0,
            kda_use_short_conv=True,
        )

    raise ValueError(f"Unknown architecture: {arch}")


def _default_hidden_candidates(arch: str, *, base_hidden_size: int) -> List[int]:
    """Generate candidate hidden sizes, respecting head-dim divisibility."""
    if arch in {"gated_deltanet", "kda"}:
        step = 64
    else:
        step = 32

    lo = max(step, int(base_hidden_size * 0.5))
    hi = int(base_hidden_size * 2.0)
    lo = (lo // step) * step
    hi = (hi // step) * step
    if lo < step:
        lo = step
    if hi < lo:
        hi = lo
    return list(range(lo, hi + 1, step))


def _arch_candidate_configs(
    arch: str,
    *,
    vocab_size: int,
    hidden_size: int,
    n_layers: int,
) -> List[ModelConfig]:
    """Generate a small candidate grid per architecture.

    Different blocks have fundamentally different parameterization; matching
    params well typically requires adjusting at least one additional knob.
    """
    if arch == "mamba2":
        # Mamba2 parameter count is quite sensitive to state size.
        candidates: List[ModelConfig] = []
        for state_size in (32, 64, 128):
            cfg = _arch_config(
                arch,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                n_layers=n_layers,
            )
            cfg.mamba2_state_size = state_size
            candidates.append(cfg)
        return candidates

    if arch == "mamba":
        # Vary state size to better match parameter budgets.
        candidates: List[ModelConfig] = []
        for state_size in (8, 16, 32):
            cfg = _arch_config(
                arch,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                n_layers=n_layers,
            )
            cfg.state_size = state_size
            candidates.append(cfg)
        return candidates

    if arch == "gated_deltanet":
        # Gated DeltaNet has extra value expansion and head_dim choices.
        candidates = []
        for head_dim in (32, 64, 128):
            for expand_v in (1.0, 1.5, 2.0):
                n_heads = max(1, hidden_size // head_dim)
                hs = n_heads * head_dim
                cfg = ModelConfig(
                    vocab_size=vocab_size,
                    hidden_size=hs,
                    n_layers=n_layers,
                    block_pattern=["gated_deltanet"],
                    gated_deltanet_heads=n_heads,
                    gated_deltanet_v_heads=n_heads,
                    gated_deltanet_head_dim=head_dim,
                    gated_deltanet_expand_v=expand_v,
                    gated_deltanet_use_short_conv=True,
                    gated_deltanet_use_gate=True,
                )
                candidates.append(cfg)
        return candidates

    if arch == "rwkv6":
        # RWKV6 has explicit FFN intermediate size and low-rank dims.
        candidates = []
        head_dim = 32
        n_heads = max(1, hidden_size // head_dim)
        hs = n_heads * head_dim
        for ffn_mult in (2, 4):
            for proj_div, gate_div in ((16, 8), (8, 4), (6, 3)):
                proj_rank = max(8, hs // proj_div)
                gate_rank = max(8, hs // gate_div)
                cfg = ModelConfig(
                    vocab_size=vocab_size,
                    hidden_size=hs,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    block_pattern=["rwkv6"],
                    rwkv6_heads=n_heads,
                    rwkv6_proj_low_rank_dim=proj_rank,
                    rwkv6_gate_low_rank_dim=gate_rank,
                    rwkv6_intermediate_size=hs * ffn_mult,
                )
                candidates.append(cfg)
        return candidates

    if arch == "kda":
        # KDA can vary head_dim and value expansion.
        candidates = []
        for head_dim in (32, 64, 128):
            for expand_v in (0.5, 1.0):
                n_heads = max(1, hidden_size // head_dim)
                hs = n_heads * head_dim
                cfg = ModelConfig(
                    vocab_size=vocab_size,
                    hidden_size=hs,
                    n_layers=n_layers,
                    n_heads=max(1, hs // max(1, hs // n_heads)),
                    head_dim=max(1, hs // n_heads),
                    block_pattern=["kda"],
                    kda_heads=n_heads,
                    kda_v_heads=n_heads,
                    kda_head_dim=head_dim,
                    kda_expand_v=expand_v,
                    kda_use_short_conv=True,
                )
                candidates.append(cfg)
        return candidates

    if arch == "deltanet":
        candidates = []
        head_dim = 32
        n_heads = max(1, hidden_size // head_dim)
        hs = n_heads * head_dim
        for expand_k in (1.0, 1.5):
            for expand_v in (1.0, 1.5, 2.0):
                cfg = ModelConfig(
                    vocab_size=vocab_size,
                    hidden_size=hs,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    block_pattern=["deltanet"],
                    deltanet_heads=n_heads,
                    deltanet_expand_k=expand_k,
                    deltanet_expand_v=expand_v,
                    deltanet_use_beta=True,
                    deltanet_use_gate=False,
                    deltanet_use_short_conv=True,
                    deltanet_chunk_size=64,
                )
                candidates.append(cfg)
        return candidates

    # Default: hidden_size-only.
    return [
        _arch_config(
            arch,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
        )
    ]


def _tune_hidden_size_for_param_budget(
    arch: str,
    *,
    vocab_size: int,
    n_layers: int,
    target_params: int,
    candidates: List[int],
) -> Tuple[ModelConfig, int]:
    """Pick the candidate hidden_size that best matches target_params."""
    best_config: Optional[ModelConfig] = None
    best_params: Optional[int] = None
    best_err: Optional[int] = None

    for hidden_size in candidates:
        for config in _arch_candidate_configs(
            arch,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
        ):
            params = _count_params_for_config(config)
            err = abs(params - target_params)
            if best_err is None or err < best_err:
                best_err = err
                best_config = config
                best_params = params

    assert best_config is not None and best_params is not None
    return best_config, best_params


def get_standardized_configs(
    vocab_size: int = 65,
    *,
    n_layers: int = 6,
    base_hidden_size: int = 256,
    reference_arch: str = "gpt",
    target_params: Optional[int] = None,
    search_hidden_sizes: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, ModelConfig]:
    """Get model configs with (approximately) matched parameter counts.

    The original benchmark used identical hyperparameters across architectures,
    which can yield large parameter-count differences due to different internal
    parameterizations. This helper instead chooses per-architecture `hidden_size`
    (and derived head counts) to match a single target parameter budget as
    closely as possible.

    Args:
        vocab_size: Vocabulary size.
        n_layers: Number of layers per architecture.
        base_hidden_size: Starting hidden size for the reference architecture.
        reference_arch: Architecture used to define the parameter budget.
        target_params: If provided, use this as the param budget instead of
            computing from the reference architecture.
        search_hidden_sizes: Optional per-arch candidate hidden sizes.

    Returns:
        Dict mapping architecture name to parameter-matched ModelConfig.
    """
    archs = ["gpt", "mamba", "mamba2", "deltanet", "gated_deltanet", "rwkv6", "rwkv7", "kda"]

    # Compute target parameter budget from reference architecture.
    if target_params is None:
        ref_cfg = _arch_config(
            reference_arch,
            vocab_size=vocab_size,
            hidden_size=base_hidden_size,
            n_layers=n_layers,
        )
        target_params = _count_params_for_config(ref_cfg)

    configs: Dict[str, ModelConfig] = {}

    for arch in archs:
        if arch == reference_arch:
            configs[arch] = _arch_config(
                arch,
                vocab_size=vocab_size,
                hidden_size=base_hidden_size,
                n_layers=n_layers,
            )
            continue

        candidates = (
            search_hidden_sizes.get(arch)
            if search_hidden_sizes is not None and arch in search_hidden_sizes
            else _default_hidden_candidates(arch, base_hidden_size=base_hidden_size)
        )

        tuned_cfg, _ = _tune_hidden_size_for_param_budget(
            arch,
            vocab_size=vocab_size,
            n_layers=n_layers,
            target_params=target_params,
            candidates=candidates,
        )
        configs[arch] = tuned_cfg

    return configs


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
