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
import os
import statistics
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import profiler

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
    profile_dir: str | None
    profile_steps: int
    profile_which: str
    profile_format: str
    perfetto_link: bool
    memory_profile: str | None
    dump_ir_dir: str | None
    dump_ir_which: str


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
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
    t1 = time.perf_counter()
    return t1 - t0


def _pctl(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile for already-sorted values."""
    if not sorted_values:
        raise ValueError("No values")
    if p <= 0:
        return sorted_values[0]
    if p >= 1:
        return sorted_values[-1]
    idx = int(round(p * (len(sorted_values) - 1)))
    return sorted_values[idx]


def _summarize_times(times: list[float]) -> dict[str, float]:
    if not times:
        raise ValueError("No times to summarize")
    values = sorted(times)
    return {
        "avg": statistics.fmean(values),
        "median": statistics.median(values),
        "min": values[0],
        "p90": _pctl(values, 0.90),
        "max": values[-1],
    }


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

    # Optional IR dump (helps map TPU trace "custom-call.*" to real ops)
    if cfg.dump_ir_dir is not None:
        dump_dir = os.path.abspath(cfg.dump_ir_dir)
        os.makedirs(dump_dir, exist_ok=True)
        print("\n--- IR dump ---")
        print(f"dir: {dump_dir} | which: {cfg.dump_ir_which}")

        def _dump(name: str, lowered):
            # Prefer StableHLO (text MLIR) which is usually available.
            text = None
            try:
                text = lowered.as_text()
            except Exception:
                pass
            if text is None:
                # Fallback: try compiler_ir (API varies by JAX version)
                try:
                    ir = lowered.compiler_ir()
                    text = str(ir)
                except Exception:
                    text = "<unable to dump IR>"
            out_path = os.path.join(dump_dir, f"{name}.mlir")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"wrote: {out_path}")

            # Print a quick hint if custom calls exist.
            if "custom_call" in text or "custom-call" in text:
                print(f"hint: '{name}' IR contains custom calls (search for 'custom_call')")

        if cfg.dump_ir_which in ("recurrent", "both"):
            lowered = rec_jit.lower(q, k, v, g, beta)
            _dump("kda_recurrent", lowered)
        if cfg.dump_ir_which in ("chunkwise", "both"):
            lowered = chk_jit.lower(q, k, v, g, beta)
            _dump("kda_chunkwise", lowered)

    print("\n=== KDA micro-bench ===")
    print(f"device: {jax.devices()[0]}")
    print(
        "shape: "
        f"B={cfg.batch} H={cfg.heads} T={cfg.seq_len} K={cfg.key_dim} V={cfg.value_dim} "
        f"chunk={cfg.chunk_size} dtype={cfg.dtype} l2norm={cfg.use_qk_l2norm}"
    )

    # Compile-only and first-exec timings.
    # Note: JAX compilation caches by computation signature; once compiled, the
    # subsequent call should execute without recompiling.
    t_compile_only_rec = _time_one(lambda: rec_jit.lower(q, k, v, g, beta).compile())
    t_first_exec_rec = _time_one(rec_jit, q, k, v, g, beta)

    t_compile_only_chk = _time_one(lambda: chk_jit.lower(q, k, v, g, beta).compile())
    t_first_exec_chk = _time_one(chk_jit, q, k, v, g, beta)

    print(f"compile-only recurrent: {t_compile_only_rec*1e3:.2f} ms")
    print(f"1st exec recurrent:     {t_first_exec_rec*1e3:.2f} ms")
    print(f"compile+1st recurrent:  {(t_compile_only_rec + t_first_exec_rec)*1e3:.2f} ms")

    print(f"compile-only chunkwise: {t_compile_only_chk*1e3:.2f} ms")
    print(f"1st exec chunkwise:     {t_first_exec_chk*1e3:.2f} ms")
    print(f"compile+1st chunkwise:  {(t_compile_only_chk + t_first_exec_chk)*1e3:.2f} ms")

    # Warmup
    for _ in range(cfg.warmup):
        _ = rec_jit(q, k, v, g, beta)
        _ = chk_jit(q, k, v, g, beta)
    jax.block_until_ready(rec_jit(q, k, v, g, beta)[0])
    jax.block_until_ready(chk_jit(q, k, v, g, beta)[0])

    # Timed loops
    rec_times = [_time_one(rec_jit, q, k, v, g, beta) for _ in range(cfg.iters)]
    chk_times = [_time_one(chk_jit, q, k, v, g, beta) for _ in range(cfg.iters)]

    rec_stats = _summarize_times(rec_times)
    chk_stats = _summarize_times(chk_times)

    print("\n--- steady-state ---")
    print(
        "recurrent: "
        f"avg {rec_stats['avg']*1e3:.3f} ms | "
        f"med {rec_stats['median']*1e3:.3f} ms | "
        f"p90 {rec_stats['p90']*1e3:.3f} ms | "
        f"min {rec_stats['min']*1e3:.3f} ms | "
        f"max {rec_stats['max']*1e3:.3f} ms | "
        f"{tokens/rec_stats['avg']:,.0f} tok/s"
    )
    print(
        "chunkwise: "
        f"avg {chk_stats['avg']*1e3:.3f} ms | "
        f"med {chk_stats['median']*1e3:.3f} ms | "
        f"p90 {chk_stats['p90']*1e3:.3f} ms | "
        f"min {chk_stats['min']*1e3:.3f} ms | "
        f"max {chk_stats['max']*1e3:.3f} ms | "
        f"{tokens/chk_stats['avg']:,.0f} tok/s"
    )
    print(f"speedup (rec/chunk): {rec_stats['avg']/chk_stats['avg']:.2f}x")

    # Optional profiler trace (TensorBoard-compatible)
    if cfg.profile_dir is not None:
        trace_dir = os.path.abspath(cfg.profile_dir)
        os.makedirs(trace_dir, exist_ok=True)
        steps = max(1, int(cfg.profile_steps))
        which = cfg.profile_which
        print(f"\n--- profiling ---")
        print(f"trace_dir: {trace_dir}")
        print(f"steps: {steps} | which: {which}")
        if cfg.profile_format == "tensorboard":
            print("view: tensorboard (optional)")
        else:
            print("view: perfetto (no TensorBoard needed)")
            print("Tip: open https://ui.perfetto.dev and upload the trace file.")
            if cfg.perfetto_link:
                print("NOTE: --perfetto-link will BLOCK until opened.")

        def _profile_loop(name: str, fn):
            # Warm one iteration to ensure executable is ready
            _ = fn(q, k, v, g, beta)

            subdir = os.path.join(trace_dir, name)
            os.makedirs(subdir, exist_ok=True)

            if cfg.profile_format == "perfetto":
                # create_perfetto_trace writes an additional Perfetto trace file.
                # create_perfetto_link prints a link and blocks until opened.
                ctx = profiler.trace(
                    subdir,
                    create_perfetto_trace=True,
                    create_perfetto_link=cfg.perfetto_link,
                )
            else:
                # TensorBoard format trace
                ctx = None

            if ctx is None:
                profiler.start_trace(subdir)
                try:
                    for i in range(steps):
                        with profiler.StepTraceAnnotation(f"{name}_step", step_num=i):
                            out = fn(q, k, v, g, beta)
                            out0 = out[0] if isinstance(out, tuple) else out
                            out0.block_until_ready()
                finally:
                    profiler.stop_trace()
            else:
                with ctx:
                    for i in range(steps):
                        with profiler.StepTraceAnnotation(f"{name}_step", step_num=i):
                            out = fn(q, k, v, g, beta)
                            out0 = out[0] if isinstance(out, tuple) else out
                            out0.block_until_ready()

        if which in ("recurrent", "both"):
            _profile_loop("kda_recurrent", rec_jit)
        if which in ("chunkwise", "both"):
            _profile_loop("kda_chunkwise", chk_jit)

        if cfg.memory_profile is not None:
            mem_path = os.path.abspath(cfg.memory_profile)
            print(f"\n--- memory profile ---")
            print(f"writing: {mem_path}")
            profiler.save_device_memory_profile(mem_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--key-dim", type=int, default=64)
    p.add_argument("--value-dim", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--no-qk-l2norm", action="store_true", help="Disable q/k l2norm in kernel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="If set, write a short JAX profiler trace to this directory (view with TensorBoard).",
    )
    p.add_argument(
        "--profile-steps",
        type=int,
        default=20,
        help="Number of profiled iterations per kernel.",
    )
    p.add_argument(
        "--profile-which",
        type=str,
        default="both",
        choices=["recurrent", "chunkwise", "both"],
        help="Which kernel(s) to profile.",
    )
    p.add_argument(
        "--profile-format",
        type=str,
        default="perfetto",
        choices=["perfetto", "tensorboard"],
        help="Trace output format. 'perfetto' works without TensorBoard.",
    )
    p.add_argument(
        "--perfetto-link",
        action="store_true",
        help=(
            "Create and print a Perfetto UI link (BLOCKS until opened). "
            "Not recommended for Kaggle notebooks."
        ),
    )
    p.add_argument(
        "--memory-profile",
        type=str,
        default=None,
        help="If set, write a device memory profile to this file.",
    )
    p.add_argument(
        "--dump-ir-dir",
        type=str,
        default=None,
        help="If set, dump lowered StableHLO/MLIR for the jitted kernels into this directory.",
    )
    p.add_argument(
        "--dump-ir-which",
        type=str,
        default="both",
        choices=["recurrent", "chunkwise", "both"],
        help="Which kernel(s) to dump IR for.",
    )
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
        profile_dir=args.profile_dir,
        profile_steps=args.profile_steps,
        profile_which=args.profile_which,
        profile_format=args.profile_format,
        perfetto_link=args.perfetto_link,
        memory_profile=args.memory_profile,
        dump_ir_dir=args.dump_ir_dir,
        dump_ir_which=args.dump_ir_which,
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
