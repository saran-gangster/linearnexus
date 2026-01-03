# Testing LinearNexus

This repo uses **pytest** for tests and includes a fairly comprehensive CPU test suite.

## Quick start

From the repo root:

```bash
# Install core + dev tooling (pytest, coverage, black, ruff)
pip install -e ".[dev]"

# Run all tests
python -m pytest -q
```

## What “test everything” means in this repo

The test suite in `tests/` covers:

- **Core utilities** (e.g. causal depthwise conv, cache/state helpers)
- **Blocks and kernels** (attention, Mamba/Mamba2, MLA, DeltaNet, KDA, RWKV6/7)
- **End-to-end smoke** for models, training loops, generation, and checkpoint save/load

If you want a single file that exercises most components, run:

```bash
python -m pytest -q tests/test_comprehensive.py
```

Note: `tests/test_comprehensive.py` explicitly forces CPU via `jax.config.update("jax_platform_name", "cpu")`.

## Installing the right dependencies

### Recommended (stable) install

```bash
pip install -e .
pip install -e ".[dev]"
```

### Flax/JAX compatibility note

This codebase uses **Flax NNx**, which is sensitive to Flax/JAX version skew.

The project currently targets Flax **0.12.1–0.12.2**.

If you see NNx errors about containers being treated as static (e.g. mentioning a static attribute like `blocks`), it usually means a Python `list`/`dict` holding NNx Modules/Arrays needs to be stored in an NNx container (e.g. `nnx.List`) instead.

## Running tests

### Full test suite

```bash
python -m pytest -q
```

### A single test file

```bash
python -m pytest -q tests/test_mla.py
```

### A single test (by keyword)

```bash
python -m pytest -q -k "rwkv7" 
```

### Show print output / debug locally

```bash
python -m pytest -s -q
```

### Stop on first failure

```bash
python -m pytest -x
```

### Run with more detail

```bash
python -m pytest -v
```

## Coverage

If you have `pytest-cov` installed (it’s included in `.[dev]`), you can run:

```bash
python -m pytest --cov=linearnexus --cov-report=term-missing
```

To generate an HTML report:

```bash
python -m pytest --cov=linearnexus --cov-report=html
# open htmlcov/index.html
```

## Formatting and linting

### Ruff (lint)

```bash
python -m ruff check .
```

### Black (format)

```bash
python -m black .
```

To only check formatting (no changes):

```bash
python -m black --check .
```

Note: The repo currently includes many files under `examples/fla/` that are not Black-formatted, so `black --check .` will report many diffs until formatted.

## Common issues

### Tests run on GPU when you expected CPU

Some tests may use your default JAX backend. If you want to force CPU for a test run:

```bash
JAX_PLATFORM_NAME=cpu python -m pytest -q
```

### You’re on GPU and see CUDA/XLA errors

Try forcing CPU (above) to isolate whether it’s a backend/driver issue vs a correctness issue.

### Very slow tests

Some tests perform forward+backward passes and can be slow on CPU.

Pragmatic workflow:

```bash
# Fast signal for a local change
python -m pytest -q tests/test_<area_you_changed>.py

# Full regression
python -m pytest -q
```
