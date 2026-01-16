#!/usr/bin/env python3
"""Benchmark Delta Rule kernels: reference vs Triton backend.

Runs on CPU or GPU. On CPU-only machines, the Triton backend will fall back
 to the reference implementation, so timings should be similar.
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

from linearnexus.kernels.backend import backend_availability
from linearnexus.modules.linear_attn.delta_net import delta_rule_recurrent


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
    if isinstance(out, tuple):
        for x in out:
            if hasattr(x, "block_until_ready"):
                x.block_until_ready()
    else:
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
    t1 = time.perf_counter()
    return t1 - t0


def _summarize(times: list[float]) -> dict[str, float]:
    values = sorted(times)
    return {
        "avg": statistics.fmean(values),
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
    }


def run_bench(
    batch: int,
    heads: int,
    seq_len: int,
    key_dim: int,
    value_dim: int,
    warmup: int,
    iters: int,
    dtype: str,
    seed: int,
) -> None:
    dtype = _dtype_from_str(dtype)
    key = jax.random.PRNGKey(seed)

    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (batch, heads, seq_len, key_dim), dtype=dtype)
    k = jax.random.normal(k2, (batch, heads, seq_len, key_dim), dtype=dtype)
    v = jax.random.normal(k3, (batch, heads, seq_len, value_dim), dtype=dtype)
    beta = jax.random.uniform(k4, (batch, heads, seq_len), minval=0.0, maxval=1.0, dtype=jnp.float32)

    ref_fn = jax.jit(lambda q, k, v, beta: delta_rule_recurrent(q, k, v, beta, backend="reference"))
    tri_fn = jax.jit(lambda q, k, v, beta: delta_rule_recurrent(q, k, v, beta, backend="triton"))

    print("\n=== Delta Rule benchmark ===")
    print(f"device: {jax.devices()[0]}")
    print(f"shape: B={batch} H={heads} T={seq_len} K={key_dim} V={value_dim} dtype={dtype}")
    avail = backend_availability()
    print(f"triton available: {avail.triton}")

    # Warmup (compile + first run)
    print("\n[reference] compile + first run")
    t_ref_first = _time_one(ref_fn, q, k, v, beta)
    print(f"time: {t_ref_first:.6f}s")

    print("\n[triton] compile + first run")
    t_tri_first = _time_one(tri_fn, q, k, v, beta)
    print(f"time: {t_tri_first:.6f}s")

    # Steady-state timing
    ref_times = []
    tri_times = []
    for _ in range(warmup):
        _time_one(ref_fn, q, k, v, beta)
        _time_one(tri_fn, q, k, v, beta)

    for _ in range(iters):
        ref_times.append(_time_one(ref_fn, q, k, v, beta))
    for _ in range(iters):
        tri_times.append(_time_one(tri_fn, q, k, v, beta))

    ref_stats = _summarize(ref_times)
    tri_stats = _summarize(tri_times)

    tokens = batch * seq_len
    print("\n[reference] steady-state")
    print(f"avg: {ref_stats['avg']:.6f}s | tokens/s: {tokens / ref_stats['avg']:.2f}")

    print("\n[triton] steady-state")
    print(f"avg: {tri_stats['avg']:.6f}s | tokens/s: {tokens / tri_stats['avg']:.2f}")

    # Quick correctness check
    y_ref, s_ref = ref_fn(q, k, v, beta)
    y_tri, s_tri = tri_fn(q, k, v, beta)
    max_diff = jnp.max(jnp.abs(y_ref - y_tri)).item()
    state_diff = jnp.max(jnp.abs(s_ref - s_tri)).item()
    print("\n[diff] max |output| diff:", max_diff)
    print("[diff] max |state| diff:", state_diff)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Delta Rule reference vs Triton kernels.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_bench(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        warmup=args.warmup,
        iters=args.iters,
        dtype=args.dtype,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
