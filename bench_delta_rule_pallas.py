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


def _summarize_error(y_ref: jax.Array, y_test: jax.Array) -> str:
    y_ref_f32 = y_ref.astype(jnp.float32)
    y_test_f32 = y_test.astype(jnp.float32)
    diff = jnp.abs(y_ref_f32 - y_test_f32)

    nan_count = int(jnp.sum(jnp.isnan(y_test_f32)))
    inf_count = int(jnp.sum(jnp.isinf(y_test_f32)))
    finite = bool(jnp.all(jnp.isfinite(y_test_f32)))

    # Use nan-safe max so we still get a signal when NaNs are present.
    max_abs = float(jnp.nanmax(diff))
    mean_abs = float(jnp.nanmean(diff))
    return (
        f"finite={finite} nan={nan_count} inf={inf_count} "
        f"max_abs_err={max_abs:.3e} mean_abs_err={mean_abs:.3e}"
    )


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
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run a preset sweep of larger shapes (GPU recommended).",
    )
    args = p.parse_args()

    dtype = {"bf16": jnp.bfloat16, "fp16": jnp.float16, "fp32": jnp.float32}[args.dtype]

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    ref = jax.jit(
        lambda q, k, v, beta: delta_rule_recurrent(q, k, v, beta, backend="reference")[0]
    )
    pallas = jax.jit(
        lambda q, k, v, beta: delta_rule_recurrent(q, k, v, beta, backend="pallas")[0]
    )

    if args.sweep:
        tests = [
            # (batch, heads, seq_len, key_dim, value_dim)
            (args.batch, args.heads, 256, args.key_dim, args.value_dim),
            (args.batch, args.heads, 512, args.key_dim, args.value_dim),
            (args.batch, args.heads, 1024, args.key_dim, args.value_dim),
            (args.batch, args.heads, 2048, args.key_dim, args.value_dim),
        ]
    else:
        tests = [(args.batch, args.heads, args.seq_len, args.key_dim, args.value_dim)]

    if args.sweep and not _has_gpu():
        print("pallas: skipped (no GPU) — sweep still runs reference")

    for (b, h, t, kd, vd) in tests:
        kk1, kk2, kk3, kk4 = jax.random.split(jax.random.key(0), 4)
        q = jax.random.normal(kk1, (b, h, t, kd), dtype)
        k = jax.random.normal(kk2, (b, h, t, kd), dtype)
        v = jax.random.normal(kk3, (b, h, t, vd), dtype)
        # beta in [0, 1), float32 (mirrors training usage).
        beta = jax.random.uniform(kk4, (b, h, t), jnp.float32)

        print(f"\nshape: batch={b} heads={h} seq_len={t} key_dim={kd} value_dim={vd} dtype={args.dtype}")
        ref_time = _bench(ref, q, k, v, beta, warmup=args.warmup, iters=args.iters)
        print(f"reference: {ref_time * 1e3:.3f} ms/iter")

        if not _has_gpu():
            continue

        try:
            pallas_time = _bench(pallas, q, k, v, beta, warmup=args.warmup, iters=args.iters)
        except Exception as e:
            print(f"pallas: failed ({type(e).__name__}: {e})")
            continue

        # Correctness check (one shot)
        y_ref = ref(q, k, v, beta)
        y_pallas = pallas(q, k, v, beta)
        err_summary = _summarize_error(y_ref, y_pallas)
        print(f"pallas:    {pallas_time * 1e3:.3f} ms/iter ({err_summary})")


if __name__ == "__main__":
    main()
