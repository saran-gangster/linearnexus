# LinearNexus

High-performance linear and hybrid attention research stack built on JAX + Pallas, targeting GPU **and** TPU from day one.

---

## Why LinearNexus

| Limitation in existing stacks (e.g., flash-linear-attention) | LinearNexus response |
| --- | --- |
| Triton-only kernels → GPU lock-in and diminishing gains vs hand-tuned CUDA | Pallas kernels that lower to Mosaic (GPU/TPU) and Triton automatically |
| PyTorch-centric training loop → limited leverage of JAX transformations | Native JAX design: jit/vmap/pjit-first with Flax NNx/Optax |
| Fragmented implementations per attention variant | Protocol-based kernel architecture + centralized core utilities |
| Scattered cross-cutting code (caching, padding, gating) across features | Unified `linearnexus/core/` runtime library with composable helpers |
| No explicit feature registry → manual glue for tests/benchmarks/docs | `linearnexus/registry.py` enables automated tooling and documentation |

We are building an **execution substrate** for linear attention research that feels as productive as high-level JAX while matching the throughput of bespoke CUDA. The first milestone centers on custom, paper-faithful Mamba/Mamba2 selective state-space kernels, implemented in a way that makes swapping in future mechanisms trivial.

---

## Product Pillars

1. **Performance** – Protocol-based Pallas kernels (chunk / fused chunk / recurrent) with software pipelining, warp specialization, and tensor-core aware tiling.
2. **Portability** – Single codebase targeting GPU (Mosaic/Triton) and TPU (Mosaic) via Pallas, plus pure JAX fallbacks.
3. **Modularity** – Core runtime library (`cache`, `padding`, `gating`, `conv`) eliminates duplication; explicit registries enable automated testing/docs.
4. **Productivity** – NNx state threading, automatic mode selection, and shared test harnesses let researchers focus on algorithms, not plumbing.

---

## Scope (Wave 1)

| In scope | Out (future) |
| --- | --- |
| Core runtime library (cache, padding, gating, conv) | Multi-node distributed training |
| Registry system for kernels/layers/models + automated test harness | Large-scale hyperparameter sweeps |
| Custom Mamba/Mamba2 kernels + modular selective-SSM layers | RetNet/GLA/DeltaNet ports (gated on compute) |
| Training pipeline (Flax NNx/Optax) for single GPU/CPU | UI / experiment tracking bundles |
| Benchmark + profiling harness for constrained hardware (Colab/12GB GPUs) | Hosted inference endpoints |
| Authoritative docs + reproducible low-compute perf baselines | Pretrained model zoo |

---

## Architecture at a Glance

```
┌───────────┐   kernels/*            (Protocol-based: Reference JAX → Pallas GPU/TPU)
│  Kernels  │   Implements SelectiveKernelProtocol with chunk + recurrent modes
└────┬──────┘
     │ Pure functional JAX Array interfaces
┌────▼──────┐   core/*               (Shared runtime: cache, padding, gating, conv)
│   Core    │   Cross-cutting utilities reused by all layers
└────┬──────┘
     │ Shape transforms, state management
┌────▼──────┐   layers/*.py          (NNx modules: projections + kernel wiring)
│  Layers   │   Mode selection, mask handling, autoregressive caching
└────┬──────┘
     │ Flax NNx module composition
┌────▼──────┐   models/*.py          (Transformer blocks, LM heads, configs)
│  Models   │   Training/inference entry points
└────┬──────┘
     │ Experiment orchestration
┌────▼──────┐   registry.py + tooling/  (Feature registry, tests, benchmarks)
│ Tooling   │   Automated parity tests, docs generation, profiling
└───────────┘
```

- **Data flow**: packed sequences → chunked kernel invocations → recurrent state handoff → metrics/logs.
- **Extension**: add a kernel implementation, wrap it with a layer strategy, register config → instantly train/eval.

For deeper details see `ARCHITECTURE.md`.

---

## Execution Snapshot

| Phase | Window | Primary outcomes | Exit criteria |
| --- | --- | --- | --- |
| Phase 0 – Proof-of-Runway | Weeks 0-2 | Tiny Mamba block + selective kernel protocol + runnable notebook | `poetry run pytest` + `python examples/train_mamba_tiny.py` succeed locally |
| Phase 1 – Paper-Faithful MVP | Weeks 3-6 | Pallas-backed selective SSM kernel + docs + Colab demo | Public Colab badge w/ TinyShakespeare curve |
| Phase 2 – Research Hardening | Weeks 7-12 | Instrumentation, ablations, modular API polish | Report + tagged release + contributor guide |
| Phase 3 – Scale-Out Prep (Gated) | Post-compute | Resume multi-mechanism roadmap once hardware secured | Funding/sponsorship agreement signed |

Full milestone details live in `ROADMAP.md`.

---

## Getting Started (planned flow)

1. Install JAX + hosting runtime (CUDA 12 / ROCm / TPU).
2. `pip install -e .[dev]` – wires in Flax/Optax/einops + tooling.
3. Run `make smoke` (fast kernel correctness) then `python benchmarks/run.py --preset constrained`.
4. Launch sample training: `python examples/train_mamba_tiny.py --config configs/models/mamba_tiny.yaml`.

Scripts/configs are stubbed until kernels land; keep an eye on Issues for progress.

### NNx Mamba Layer

Wave 1 now includes a paper-faithful, CPU-friendly selective SSM implementation powered by Flax NNx:

- `linearnexus/layers/mamba.py` wires projections + caching to the reference selective scan kernel (`linearnexus/kernels/mamba_reference.py`).
- **Pallas GPU backend status**: `linearnexus/kernels/mamba_pallas.py` exists but is **currently disabled** (Phase 0 constraint). Triton/Pallas does not support `dynamic_slice` or sequential `lax.scan` operations required for Mamba's selective scan. Phase 1+ will implement a handwritten Triton kernel bypassing the Pallas abstraction.
- Pick a backend via `MambaConfig.kernel_backend` (`"reference"` [default], `"pallas"` [raises NotImplementedError], or `"auto"`).
- `tests/test_mamba_layer.py` keeps the chunked and recurrent paths numerically aligned, while `tests/test_mamba_kernels.py` (currently disabled) will test GPU parity in Phase 1+.
- `examples/run_mamba_reference.py` remains a tiny smoke test using the reference kernel.

Try it locally once dependencies are installed:

```bash
pytest tests/test_mamba_layer.py                   # CPU tests pass
# pytest tests/test_mamba_kernels.py               # Disabled (Pallas unsupported in Phase 0)
python examples/run_mamba_reference.py --batch 2 --seq 16 --hidden 64
```

---

## Contribution Expectations

- Use type hints + thorough docstrings for any public surface.
- Pair every kernel change with: reference implementation parity test, perf sample (`benchmarks/kernel_*.py`).
- Keep architectural decisions in `ARCHITECTURE.md` up to date (PR must include note if the change touches kernel/layer contracts).

---

## References

- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) – Triton baseline.
- [JAX Pallas Design Notes](https://docs.jax.dev/en/latest/pallas/design/design.html) and [Mosaic GPU pipelining](https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html).
- Mamba (ICLR 2024), Mamba 2 (2025), RetNet (ICLR 2024) for contrast, plus GLA/DeltaNet as future targets.

---

**Status**: Planning | **Last updated**: 18 Nov 2025

For questions or collaboration requests please open a GitHub issue with the `discussion` label.
