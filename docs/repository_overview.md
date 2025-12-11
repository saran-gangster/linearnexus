# LinearNexus — Repository Overview & Developer Guide

This document summarizes the key architecture, files, and recommended development workflows for the LinearNexus repository. It focuses on key design decisions, module responsibilities, shape conventions, running and test steps, and suggestions for future work.

---

## 📦 Repository Structure (High-level)

- `linearnexus/` - Core library package
  - `models.py` — `ModelConfig`, `create_model`, `LMModel` (model factory and block wiring)
  - `generate.py` — Sampling, generation loops, token streaming
  - `optim.py` — Optimizers and schedules (AdamW, Muon, Sophia)
  - `data.py` — Tokenizers, `TextDataset`, `DataLoader`
  - `train/` — Trainers (SFT, GRPO, PPO) and training utilities
  - `modules/` — Implementation of building blocks
    - `attention/` — `causal.py` (`CausalSelfAttention`, `AttentionBlock`, KV caching) and `mla.py` (MLA and low-rank KV cache)
    - `ssm/` — `mamba.py` (`MambaBlock` and `selective_scan_ref`)
    - `common.py` — shared NNx helpers: `Embedding`, `RMSNorm`, `MLP`, `RotaryEmbedding`
  - `core/` — shared low-level helpers: conv, cache, padding, gating, mode, config
  - `kernels/` — kernel implementations (e.g., `mamba_reference.py`)
- `docs/` — Developer documentation and guides
- `tests/` — Unit tests (e.g., `tests/test_mla.py`)
- `train_lm.py`, `sample.py` — Example CLI tools for training and generation
- `pyproject.toml` — Project metadata and dependencies

---

## 🧭 Architecture Overview

- **Block Protocol**: Every block in `modules/` implements the same interface:
  ```python
  def __call__(self, x: jax.Array, *, state: Optional[BlockState] = None, mask: Optional[jax.Array] = None, mode: Optional[str] = None) -> tuple[jax.Array, Optional[BlockState]]:
      ...
  ```
  This allows the `LMModel` to iterate blocks uniformly, enabling GPT, Mamba (SSM), or hybrid models via `ModelConfig.block_pattern`.

- **Model Composition**: `LMModel` uses `create_model` / `ModelConfig` presets (like `GPT_SMALL`, `MAMBA_SMALL`), then instantiates blocks in `_create_block`. The `blocks` list controls the model graph.

- **Modes**:
  - `chunk` mode (parallel chunked processing) — used in training.
  - `recurrent` mode (token-by-token) — used for generation and efficient inference with caching.
  - `KernelMode` in `core/mode.py` and kernels may select mode automatically.

- **State Management**:
  - Attention: `KVCache` stores `keys`, `values`, and `position` (see `modules/attention/causal.py`).
  - Mamba SSM: `MambaState` bundles `conv_state`, `ssm_state`, `position` (see `modules/ssm/mamba.py`).

---

## 🔬 Core Modules & Key Files

- `linearnexus/core/` (Utilities)
  - `conv.py` — `depthwise_conv1d_causal(inputs, weight, bias, cache=None)` — depthwise conv + cache semantics
  - `cache.py` — `ConvState`, `RecurrentState` helper wrappers; `@classmethod zeros` to initialize
  - `padding.py` — `pad`, `unpad`, `compute_unpadded_indices` for packed sequences
  - `gating.py` — `low_rank_project`, `normalize_gate_logits` used by selective SSM & gating logic
  - `mode.py` — `KernelMode` enum and `select_mode` to decide chunk threshold

- `linearnexus/kernels/mamba_reference.py` — Reference selective SSM kernel with:
  - `MambaReferenceKernel(..., mode)` with `forward_chunk(...)` and `forward_recurrent(...)` implemented using `jax.lax.scan()`.
  - Dataclasses: `MambaKernelParams`, `MambaKernelInputs`, `MambaKernelState`.

- `linearnexus/modules/ssm/mamba.py` — `MambaBlock`:
  - Input norm, in-projection -> `hidden`, `gate` split
  - `depthwise_conv1d_causal` -> activation -> `x_proj` -> `time_step`, `B`, `C`
  - `delta` is computed via `dt_proj` + `softplus`
  - `selective_scan_ref` is a chunked `lax.scan` implementation enforcing correct sequential state.
  - Output projection and final state update

- `linearnexus/modules/attention/causal.py` — `CausalSelfAttention` and `AttentionBlock`:
  - Q/K/V projections; RoPE optional
  - `KVCache` for generation with `update()` and `get()` utilities
  - Supports MHA, GQA, MQA via `n_kv_heads` and `n_rep` repetition
  - Causal masking with offset for cached tokens

- `linearnexus/modules/attention/mla.py` — `MultiHeadLatentAttention` and `MLABlock`:
  - Low-rank compression to reduce KV cache memory
  - `MLACache` caches compressed KV and RoPE keys
  - Works analogously to `AttentionBlock` but with compressed KV storage

- `linearnexus/models.py` — `LMModel` and `ModelConfig`:
  - `ModelConfig` defines the architecture (vocab, hidden size, block pattern)
  - `create_model` returns a `ModelConfig` or an instantiated `LMModel`
  - `LMModel.__call__()` wires the blocks, does final normalization and projection
  - `init_state(batch_size)` constructs `ModelState` with per-block states depending on the block type

- `linearnexus/train/` — Trainer modules
  - `train/sft.py` — `SFTTrainer`: `loss_fn`, `train_step` (jit decorated), training loop, checkpointing using `nnx` state conversions

---

## ✅ Running & Development Workflow (Quick Start)

1. Setup & install (local dev mode):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

2. Run unit tests:

```bash
pytest -q
# Example single-file test:
pytest tests/test_mla.py -q
```

3. Train a small model (quick):

```bash
python train_lm.py --model gpt-small --download-shakespeare --max-steps 100 --batch-size 2
```

4. Generate text:

```bash
python sample.py --checkpoint checkpoints/step_100 --prompt "Hello" --max-tokens 50
```

---

## 🧪 Testing & Parity

- Unit tests include `tests/test_mla.py`. The library contains other test scaffolding in `tests/conftest.py`.
- **Suggested parity tests** to add:
  - `MambaBlock` parity vs `MambaReferenceKernel` (chunk vs recurrent, rtol=1e-4)
  - `selective_scan_ref` vs `MambaReferenceKernel` (parameters, inputs, outputs)
  - Add chunk vs recurrent parity tests for `MambaBlock` and `CausalSelfAttention` where caching paths are used.

---

## 🔧 Developer Notes & Best Practices

- Shape conventions are important:
  - Public block inputs/outputs: `[batch, seq, hidden]`.
  - Kernels and SSM internals often use `[batch, intermediate, seq]` or `[batch, seq, state]`; pay attention to transposes in `mamba.py` and `mamba_reference.py`.

- Parameter objects (`nnx.Param`) store values in `.value`; access as `self.conv_weight.value`.
- For reproducibility and easier debugging, you can disable JIT:
  ```python
  import jax
  jax.config.update('jax_disable_jit', True)
  ```
- Dtype management: Many kernels cast to `float32`. Be careful when experimenting with mixed precision.
- Use existing helper classes for state initialization: `KVCache.zeros`, `MambaState.zeros`, `ConvState.zeros`.

---

## 🚀 Suggestions for Future Work (Priority aware)

- Add explicit parity tests between `selective_scan_ref` and `MambaReferenceKernel` to make kernel replacement safe.
- Add Pallas/Triton kernels behind a kernel adapter interface and keep pure JAX reference implementation for correctness and tests.
- Improve checkpoint storage format to `np.save` or `npz` for large arrays and add a `load` routine to support exact parameter reload.
- Expand `tests/` coverage for `MambaBlock` chunking behavior, `CausalSelfAttention` KV cache correctness, and trainer integration.
- Add GitHub Actions/CI `pytest` run and `flake8/ruff` checks to enforce code style and tests on PRs.

---

## Resources & How To Extend

- To **add a new block**: follow `docs/adding_new_blocks.md` (already in repo). Key steps:
  - Implement `YourBlock(nnx.Module)` with `__call__` and `init_state`
  - Add exports in `modules/your_block/__init__.py`
  - Register block type in `_create_block` of `models.py` and in `ModelConfig` if needed
  - Add tests under `tests/` to validate shape, behavior, and caching

- To **add a new optimizer or training mode**:
  - Create the optimizer in `linearnexus/optim.py` (conform to `optax.GradientTransformation`)
  - Hook into `create_optimizer` and add CLI options in `train_lm.py`

---

## Wrap-Up

This document highlights the major points of the codebase structure, patterns, shape conventions, and practical next steps. If you'd like, I can follow up by:
- Creating parity tests for the `Mamba` kernel vs reference
- Running the test suite and reporting failures
- Adding a small, runnable notebook example (CPU-only) that trains a tiny model end-to-end

Which of the above would you like to focus on next? Feel free to ask for any expansions or additions to this doc.
