#!/usr/bin/env python3
"""Micro-benchmark for KDA kernels (recurrent vs chunkwise).

Intended usage: run on your target accelerator (e.g., TPU v5 lite) and compare
steady-state performance after compilation.

This benchmarks the *kernel functions* directly:
  - kda_recurrent(q, k, v, g, beta)
  - kda_chunkwise(q, k, v, g, beta, chunk_size)

It reports:
  - compile+first-run time (includes XLA compilation)
  - steady-state average time/iter
  - tokens/sec (batch * seq_len / time)

Notes:
- KDA chunkwise expects seq_len divisible by chunk_size (it will pad internally,
  but for fair benchmarking keep it divisible).
- Inputs are generated with a stable, causal-like gate: g is negative.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from linearnexus.modules.linear_attn.kda import kda_chunkwise, kda_recurrent


@dataclass(frozen=True)
class BenchConfig:
    batch: int
    heads: int
    seq_len: int
    key_dim: int
    value_dim: int
    chunk_size: int
    warmup: int
    iters: int
    dtype: str
    use_qk_l2norm: bool
    seed: int


def _dtype_from_str(name: str) -> jnp.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return jnp.bfloat16
    if name in ("fp16", "float16"):
        return jnp.float16
    if name in ("fp32", "float32"):
        return jnp.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _time_one(fn, *args) -> float:
    t0 = time.perf_counter()
    out = fn(*args)
    # Ensure we measure device execution time
    if isinstance(out, tuple):
        for x in out:
            if hasattr(x, "block_until_ready"):
                x.block_until_ready()
    else:
        out.block_until_ready()
    t1 = time.perf_counter()
    return t1 - t0


def run_one_case(cfg: BenchConfig, *, key: jax.Array) -> None:
    dtype = _dtype_from_str(cfg.dtype)

    # Shapes match our kernels:
    # q/k/g: [B, H, T, K], v: [B, H, T, V], beta: [B, H, T]
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    q = jax.random.normal(k1, (cfg.batch, cfg.heads, cfg.seq_len, cfg.key_dim), dtype=dtype)
    k = jax.random.normal(k2, (cfg.batch, cfg.heads, cfg.seq_len, cfg.key_dim), dtype=dtype)
    v = jax.random.normal(k3, (cfg.batch, cfg.heads, cfg.seq_len, cfg.value_dim), dtype=dtype)

    # Gate is typically negative. Make it "reasonably" negative to stress stability.
    # Using float32 for g/beta is fine; kernels cast internally anyway.
    g = -jax.random.uniform(
        k4,
        (cfg.batch, cfg.heads, cfg.seq_len, cfg.key_dim),
        minval=0.0,
        maxval=10.0,
        dtype=jnp.float32,
    )
    beta = jax.random.uniform(
        k5,
        (cfg.batch, cfg.heads, cfg.seq_len),
        minval=0.0,
        maxval=1.0,
        dtype=jnp.float32,
    )

    # JIT wrappers
    rec_jit = jax.jit(lambda q, k, v, g, beta: kda_recurrent(q, k, v, g, beta, use_qk_l2norm=cfg.use_qk_l2norm))
    chk_jit = jax.jit(
        lambda q, k, v, g, beta: kda_chunkwise(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=cfg.chunk_size,
            use_qk_l2norm=cfg.use_qk_l2norm,
        )
    )

    tokens = cfg.batch * cfg.seq_len

    print("\n=== KDA micro-bench ===")
    print(f"device: {jax.devices()[0]}")
    print(
        "shape: "
        f"B={cfg.batch} H={cfg.heads} T={cfg.seq_len} K={cfg.key_dim} V={cfg.value_dim} "
        f"chunk={cfg.chunk_size} dtype={cfg.dtype} l2norm={cfg.use_qk_l2norm}"
    )

    # First-run (compile + execute)
    t_compile_rec = _time_one(rec_jit, q, k, v, g, beta)
    t_compile_chk = _time_one(chk_jit, q, k, v, g, beta)
    print(f"compile+1st recurrent: {t_compile_rec*1e3:.2f} ms")
    print(f"compile+1st chunkwise: {t_compile_chk*1e3:.2f} ms")

    # Warmup
    for _ in range(cfg.warmup):
        _ = rec_jit(q, k, v, g, beta)
        _ = chk_jit(q, k, v, g, beta)
    jax.block_until_ready(rec_jit(q, k, v, g, beta)[0])
    jax.block_until_ready(chk_jit(q, k, v, g, beta)[0])

    # Timed loops
    rec_times = [_time_one(rec_jit, q, k, v, g, beta) for _ in range(cfg.iters)]
    chk_times = [_time_one(chk_jit, q, k, v, g, beta) for _ in range(cfg.iters)]

    rec_avg = sum(rec_times) / len(rec_times)
    chk_avg = sum(chk_times) / len(chk_times)

    print("\n--- steady-state ---")
    print(f"recurrent: {rec_avg*1e3:.3f} ms/iter | {tokens/rec_avg:,.0f} tokens/s")
    print(f"chunkwise: {chk_avg*1e3:.3f} ms/iter | {tokens/chk_avg:,.0f} tokens/s")
    print(f"speedup (rec/chunk): {rec_avg/chk_avg:.2f}x")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--key-dim", type=int, default=64)
    p.add_argument("--value-dim", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--no-qk-l2norm", action="store_true", help="Disable q/k l2norm in kernel")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = BenchConfig(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        chunk_size=args.chunk_size,
        warmup=args.warmup,
        iters=args.iters,
        dtype=args.dtype,
        use_qk_l2norm=not args.no_qk_l2norm,
        seed=args.seed,
    )

    if cfg.seq_len <= 0 or cfg.key_dim <= 0 or cfg.value_dim <= 0:
        raise ValueError("seq/key/value dims must be positive")

    if cfg.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    # For best apples-to-apples, keep seq_len divisible by chunk_size.
    if cfg.seq_len % cfg.chunk_size != 0:
        print(
            f"WARNING: seq_len ({cfg.seq_len}) not divisible by chunk_size ({cfg.chunk_size}). "
            "Chunkwise will pad internally; consider using a divisible seq_len for fair timing."
        )

    print(f"JAX version: {jax.__version__}")
    print(f"backend: {jax.default_backend()}")

    key = jax.random.PRNGKey(cfg.seed)
    run_one_case(cfg, key=key)


if __name__ == "__main__":
    main()
