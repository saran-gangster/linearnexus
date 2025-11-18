# LinearNexus Development Roadmap

**Version**: 1.0  
**Last Updated**: November 18, 2025  
**Planning Horizon**: 12 months (compute-dependent)

---

## Context

LinearNexus is currently being built on extremely limited compute (single GPU / Colab tier). The immediate objective is an authentic, paper-faithful linear attention model that trains **today**, proves the stack works end-to-end, and can be shown publicly. Large-scale kernels, distributed infra, and pretrained releases remain on the roadmap but are gated on future compute access.

Key assumptions:
- MVP must run on 12GB VRAM or less and optionally on CPU for smoke tests.
- The first showcased model mirrors the Mamba / Mamba 2 selective state-space architecture but uses pared-down widths/depths.
- All future phases are conditional on securing additional hardware or sponsorship.

---

## High-Level Flight Plan

| Phase | Window | Purpose | Exit Criteria |
| --- | --- | --- | --- |
| Phase 0 – Proof-of-Runway | Weeks 0-2 | Stand up repo, configs, and toy data so a Mamba-style block trains on laptop/Colab | `poetry run pytest` + `python examples/train_mamba_tiny.py` both succeed locally |
| Phase 1 – Paper-Faithful MVP | Weeks 3-6 | Implement the full selective SSM cell (chunk + recurrent) with reference-aligned math and release a reproducible Colab notebook | Public Colab + README badge show loss/perplexity curves on TinyShakespeare (≤256 seq) |
| Phase 2 – Research Hardening | Weeks 7-12 | Add instrumentation, ablations, and lightweight benchmarks to make the MVP credible to researchers | Blog-style report + tagged release; backlog groomed for scale-up |
| Phase 3 – Scale-Out Prep (Gated) | Post-compute | Resume original multi-mechanism agenda once >4 high-memory GPUs or TPU pod slices are secured | Hardware agreement in place + budget allocated |

---

## Modularity Guardrails

To guarantee a "super modular" codebase even at small scale:
- **Layered package layout**: `kernels/`, `core/`, `layers/`, `models/` must compile without cyclic imports.
- **Core runtime library**: Shared utilities (`core/cache.py`, `core/padding.py`, `core/gating.py`, `core/conv.py`) eliminate cross-cutting code duplication.
- **Explicit registries**: `registry.py` maps feature names to kernel/layer/config classes, enabling automated testing, benchmarks, and docs.
- **Interface-first design**: Every SSM kernel conforms to `SelectiveKernelProtocol` (shape inference, chunk + recurrent entry points).
- **Swap-friendly configs**: Model configs declare components declaratively (YAML/dataclasses) so swapping Mamba↔future architectures does not touch training code.
- **Testing discipline**: Shared parity/gradient helpers (`tests/helpers/parity.py`) ensure any new module passes the same numerical correctness suites before inclusion.

These guardrails are enforced during Phase 0 and extended in later phases.

---

## Phase 0 – Proof-of-Runway (Weeks 0-2)

**Goal**: Make the repository runnable on commodity hardware and demonstrate a minimal selective State Space Model block derived from the Mamba paper.

### Deliverables
- Repository skeleton + `pyproject.toml`, linting, and smoke tests
- **Core runtime foundation**:
  - `linearnexus/core/conv.py` with reusable depthwise causal conv1d
  - `linearnexus/core/cache.py` for recurrent/conv state abstractions
  - `linearnexus/registry.py` mapping "mamba" to kernel/layer/config
- Tiny synthetic dataset loader (character-level TinyShakespeare / pile-sample)
- `linearnexus/layers/mamba.py` containing:
   - Flax NNx module with projections, depthwise conv (via `core.conv`), and selective SSM
   - Pure JAX reference kernel using `jax.lax.scan` chunking (no Pallas yet)
- Shared `SelectiveKernelProtocol` + parity tests proving chunk vs recurrent modes align
- Notebook + CLI (`examples/run_mamba_reference.py`) that runs on CPU/GPU in <10 minutes

### Compute Strategy
- Default to CPU for unit tests, optional `--use-gpu` path
- Provide Colab badge with pinned commits
- Include profiling hooks to collect wall time even on limited resources

### Success Metrics
- End-to-end training (1 epoch, batch size 4, seq len 256) completes on 12GB GPU
- Loss drops by ≥20% from initialization
- Tests run under 2 minutes on CPU

---

## Phase 1 – Paper-Faithful MVP (Weeks 3-6)

**Goal**: Deliver a custom Mamba/Mamba2 implementation that respects the architecture diagram in the original papers while staying within low-compute constraints and enforcing modular layering.

### Milestone 1.1 – Kernelization Lite (Weeks 3-4)
- Replace the reference `lax.scan` kernel with a hand-written Pallas chunk kernel that supports:
   - Adjustable chunk size (default 128 tokens)
   - Selective SSM parameters (Δ, A, B, C) exactly as defined in the papers
   - Recurrent update path for inference (still single-device)
   - Modular hooks so kernels for other SSM variants (e.g., Mamba-SSM, retaining gating) can plug in
- Implement fallbacks: automatic switch to pure JAX when GPU not detected.

### Milestone 1.2 – Training Loop + Docs (Weeks 5-6)
- **Testing infrastructure**:
  - `tests/helpers/parity.py` with reusable `assert_chunk_recurrent_parity` and `assert_mask_behavior`
  - Kernel-level tests comparing Pallas vs reference (when Pallas kernel lands)
- **Config system**: `core/config.py` with `ConfigBase` for serialization/validation; adapt `MambaConfig`
- Training script with experiment configs (YAML) for CPU, single GPU, Colab
- Colab notebook mirroring CLI and saving checkpoints to Google Drive
- Lightweight evaluation harness producing perplexity + wall-clock tables

### Public Proof
- Record run logs (tensorboard.dev or screenshots) and link them in `README.md`
- Tag release `v0.1.0-mvp` focused solely on this tiny Mamba cell

---

## Phase 2 – Research Hardening (Weeks 7-12)

**Goal**: Make the MVP persuasive, highlight modularity, and package the code so collaborators can drop in new selective SSM components once compute increases.

### Milestone 2.1 – Instrumentation & Benchmarks
- Add gradient-check tests versus NumPy reference implementation of the selective scan
- Capture per-step throughput on CPU, RTX 4090 (borrowed/spot instance), and TPU v3-8 (free TPU credits if available)
- Document memory footprint and explain tradeoffs in `TECHNICAL_REFERENCE.md`

### Milestone 2.2 – Ablations & Report
- Run 3 ablations (state dimension, Δ parametrization, convolutional skip path) on TinyShakespeare
- Summarize findings in `docs/reports/mamba_tiny_mvp.md`
- Publish blog-style write-up describing how the limited-compute work unlocks future scaling

### Milestone 2.3 – Collaboration Readiness
- **Documentation automation**:
  - Per-feature docs (`docs/features/mamba.md` with equations, modes, caveats)
  - Script to walk registry and generate feature tables for `README.md`/`ARCHITECTURE.md`
- Provide contribution guide specifically for adding new linear attention variants once compute arrives
- Draft funding/sponsorship brief highlighting what additional hardware would unlock (Mamba2 large configs, hybrid SSM+attention stacks, distributed training)

---

## Phase 3 – Scale-Out Prep (Gated on Compute)

This phase revives the original multi-mechanism roadmap (GLA, DeltaNet, hybrid layers, distributed training). Work does **not** start until one of the following is secured:
- ≥4 GPUs with ≥40GB VRAM each (cluster or sponsor)
- TPU v4-8 slices with at least 500 TPU hours committed
- Dedicated budget for cloud bursting (~$2k/month)

When triggered:
1. Re-baseline kernels with real Mosaic backends
2. Spin up benchmarking harnesses and CI for multi-device scenarios
3. Resume prior Phase 2–5 content from roadmap v1.0 (archived section)

---

## Success Metrics (Current Constraint)

| Category | Metric | Target |
| --- | --- | --- |
| Functionality | Tiny Mamba training run | <10 min on 12GB GPU, <25 min CPU |
| Evidence | Public Colab + loss/perplexity curve | Linked in README + blog |
| Quality | Unit + integration tests | 90% line coverage on `linearnexus/experimental` |
| Community | Early adopters | ≥3 external users reproduce run via Colab |

Future large-scale metrics (throughput vs flash-linear-attention, TPU utilization, model zoo) stay in the backlog but are explicitly gated.

---

## Risk Mitigation (Low-Compute Focus)

1. **Run Time Too Long on CPU**  
   *Mitigation*: keep sequence lengths ≤256, use gradient accumulation to mimic batch sizes.  
   *Contingency*: provide precomputed activations for demo, ask collaborators to run GPU tests.

2. **Pallas Kernel Requires Unsupported Hardware**  
   *Mitigation*: maintain pure JAX reference path; ensure API parity so demos never fail.  
   *Contingency*: postpone Pallas work until compute arrives, keep README honest about status.

3. **Community Perception ("Vaporware")**  
   *Mitigation*: prioritize runnable notebooks, publish logs/screenshots, keep roadmap transparent about constraints.  
   *Contingency*: focus messaging on research directions unlocked by upcoming funding.

4. **Scope Creep Before Compute**  
   *Mitigation*: change backlog label to `blocked:compute` for heavy tasks; enforce Go/No-Go review every two weeks.  
   *Contingency*: if compute still unavailable after Phase 2, pivot to theoretical research notes instead of kernel work.

---

## Review Cadence

- **Weekly sync**: Validate MVP progress, ensure demos still run on constrained hardware.
- **Bi-weekly Go/No-Go**: Decide whether to advance to next milestone or continue hardening.
- **Quarterly checkpoint**: Reassess compute status; if new resources confirmed, reintroduce archived large-scale phases with updated timelines.

---

**Document Status**: Living roadmap aligned with low-compute reality.  
**Next Review**: End of Week 2 (post Phase 0).  
**Owner**: Project Lead.

For feedback or sponsorship discussions, open a GitHub issue with the `roadmap` label or email the maintainers.
