"""Benchmark DeltaNet delta-rule recurrent: reference vs Pallas.

This benchmark is safe to run on CPU-only machines:
- It always runs the reference kernel.
- It only attempts the Pallas kernel when a GPU is present and Pallas is usable.

Usage:
  python bench_delta_rule_pallas.py --seq-len 256

You can also force backend selection via:
  LINEARNEXUS_KERNEL_BACKEND=reference python bench_delta_rule_pallas.py
  LINEARNEXUS_KERNEL_BACKEND=pallas    python bench_delta_rule_pallas.py
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from linearnexus.modules.linear_attn.delta_net import delta_rule_recurrent


def _has_gpu() -> bool:
    return any(d.platform == "gpu" for d in jax.devices())


def _bench(fn, *args, warmup: int, iters: int) -> float:
    # Warmup (compile + cache)
    for _ in range(warmup):
        fn(*args).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args).block_until_ready()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--key-dim", type=int, default=64)
    p.add_argument("--value-dim", type=int, default=64)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    dtype = {"bf16": jnp.bfloat16, "fp16": jnp.float16, "fp32": jnp.float32}[args.dtype]

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    q = jax.random.normal(k1, (args.batch, args.heads, args.seq_len, args.key_dim), dtype)
    k = jax.random.normal(k2, (args.batch, args.heads, args.seq_len, args.key_dim), dtype)
    v = jax.random.normal(k3, (args.batch, args.heads, args.seq_len, args.value_dim), dtype)
    beta = jax.random.uniform(k4, (args.batch, args.heads, args.seq_len), jnp.float32)

    ref = jax.jit(lambda q, k, v, beta: delta_rule_recurrent(q, k, v, beta, backend="reference")[0])
    ref_time = _bench(ref, q, k, v, beta, warmup=args.warmup, iters=args.iters)
    print(f"reference: {ref_time * 1e3:.3f} ms/iter")

    if not _has_gpu():
        print("pallas: skipped (no GPU)")
        return

    pallas = jax.jit(lambda q, k, v, beta: delta_rule_recurrent(q, k, v, beta, backend="pallas")[0])

    try:
        pallas_time = _bench(pallas, q, k, v, beta, warmup=args.warmup, iters=args.iters)
    except Exception as e:
        print(f"pallas: failed ({type(e).__name__}: {e})")
        return

    # Correctness check
    y_ref = ref(q, k, v, beta)
    y_pallas = pallas(q, k, v, beta)
    max_abs = jnp.max(jnp.abs(y_ref.astype(jnp.float32) - y_pallas.astype(jnp.float32)))
    print(f"pallas:    {pallas_time * 1e3:.3f} ms/iter (max_abs_err={float(max_abs):.3e})")


if __name__ == "__main__":
    main()
