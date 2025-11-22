"""Tests for TPU Mamba kernel implementation."""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from linearnexus.kernels import (
    KernelMode,
    MambaKernelInputs,
    MambaKernelParams,
    MambaKernelState,
    MambaReferenceKernel,
    MambaTPUKernel,
    TPU_AVAILABLE,
)


def _tpu_ready() -> bool:
    """Check if TPU is available for testing."""
    return TPU_AVAILABLE


@pytest.mark.skipif(not _tpu_ready(), reason="TPU kernel requires TPU device availability")
@pytest.mark.parametrize("chunk_size", [1, 4, 7])
def test_tpu_kernel_matches_reference(chunk_size: int) -> None:
    """Test that TPU kernel produces identical outputs to reference kernel."""
    batch, intermediate, seq, state_size = 2, 4, 11, 3

    # Generate random inputs with fixed seed for reproducibility
    key_hidden, key_delta, key_gate, key_B, key_C, key_a, key_d, key_state = jax.random.split(
        jax.random.PRNGKey(42), 8
    )
    hidden = jax.random.normal(key_hidden, (batch, intermediate, seq))
    delta = jax.random.normal(key_delta, (batch, intermediate, seq))
    gate = jax.random.uniform(key_gate, (batch, intermediate, seq))
    B = jax.random.normal(key_B, (batch, seq, state_size))
    C = jax.random.normal(key_C, (batch, seq, state_size))

    params = MambaKernelParams(
        a_log=jax.random.normal(key_a, (intermediate, state_size)),
        d=jax.random.normal(key_d, (intermediate,)),
    )
    inputs = MambaKernelInputs(hidden=hidden, delta=delta, B=B, C=C, gate=gate)
    init_state = MambaKernelState(ssm=jax.random.normal(key_state, (batch, intermediate, state_size)))

    # Run both kernels
    ref_kernel = MambaReferenceKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)
    tpu_kernel = MambaTPUKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)

    ref_out, ref_state = ref_kernel.forward_chunk(params, inputs, init_state, chunk_size=chunk_size)
    tpu_out, tpu_state = tpu_kernel.forward_chunk(params, inputs, init_state, chunk_size=chunk_size)

    # Verify outputs match within tolerance
    np.testing.assert_allclose(ref_out, tpu_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ref_state.ssm, tpu_state.ssm, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _tpu_ready(), reason="TPU kernel requires TPU device availability")
def test_tpu_kernel_zero_state() -> None:
    """Test TPU kernel with zero initial state."""
    batch, intermediate, seq, state_size = 1, 2, 8, 4

    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 6)
    
    hidden = jax.random.normal(keys[0], (batch, intermediate, seq))
    delta = jax.random.normal(keys[1], (batch, intermediate, seq))
    gate = jax.random.uniform(keys[2], (batch, intermediate, seq))
    B = jax.random.normal(keys[3], (batch, seq, state_size))
    C = jax.random.normal(keys[4], (batch, seq, state_size))

    params = MambaKernelParams(
        a_log=jax.random.normal(keys[5], (intermediate, state_size)),
        d=jax.random.uniform(keys[5], (intermediate,)),
    )
    inputs = MambaKernelInputs(hidden=hidden, delta=delta, B=B, C=C, gate=gate)

    # Run with None state (should initialize to zeros)
    ref_kernel = MambaReferenceKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)
    tpu_kernel = MambaTPUKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)

    ref_out, ref_state = ref_kernel.forward_chunk(params, inputs, None, chunk_size=4)
    tpu_out, tpu_state = tpu_kernel.forward_chunk(params, inputs, None, chunk_size=4)

    np.testing.assert_allclose(ref_out, tpu_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ref_state.ssm, tpu_state.ssm, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _tpu_ready(), reason="TPU kernel requires TPU device availability")
def test_tpu_kernel_recurrent_mode() -> None:
    """Test TPU kernel in recurrent mode (single token inference)."""
    batch, intermediate, state_size = 2, 4, 3

    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 7)

    # Single token (seq_len=1)
    hidden = jax.random.normal(keys[0], (batch, intermediate, 1))
    delta = jax.random.normal(keys[1], (batch, intermediate, 1))
    gate = jax.random.uniform(keys[2], (batch, intermediate, 1))
    B = jax.random.normal(keys[3], (batch, 1, state_size))
    C = jax.random.normal(keys[4], (batch, 1, state_size))

    params = MambaKernelParams(
        a_log=jax.random.normal(keys[5], (intermediate, state_size)),
        d=jax.random.uniform(keys[6], (intermediate,)),
    )
    inputs = MambaKernelInputs(hidden=hidden, delta=delta, B=B, C=C, gate=gate)
    init_state = MambaKernelState(ssm=jax.random.normal(keys[6], (batch, intermediate, state_size)))

    ref_kernel = MambaReferenceKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)
    tpu_kernel = MambaTPUKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)

    ref_out, ref_state = ref_kernel.forward_recurrent(params, inputs, init_state)
    tpu_out, tpu_state = tpu_kernel.forward_recurrent(params, inputs, init_state)

    np.testing.assert_allclose(ref_out, tpu_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ref_state.ssm, tpu_state.ssm, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _tpu_ready(), reason="TPU kernel requires TPU device availability")
def test_tpu_kernel_large_state() -> None:
    """Test TPU kernel with larger state dimension (tests padding logic)."""
    batch, intermediate, seq, state_size = 1, 2, 16, 64

    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, 7)

    hidden = jax.random.normal(keys[0], (batch, intermediate, seq))
    delta = jax.random.normal(keys[1], (batch, intermediate, seq))
    gate = jax.random.uniform(keys[2], (batch, intermediate, seq))
    B = jax.random.normal(keys[3], (batch, seq, state_size))
    C = jax.random.normal(keys[4], (batch, seq, state_size))

    params = MambaKernelParams(
        a_log=jax.random.normal(keys[5], (intermediate, state_size)),
        d=jax.random.uniform(keys[6], (intermediate,)),
    )
    inputs = MambaKernelInputs(hidden=hidden, delta=delta, B=B, C=C, gate=gate)
    init_state = MambaKernelState(ssm=jax.random.normal(keys[6], (batch, intermediate, state_size)))

    ref_kernel = MambaReferenceKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)
    tpu_kernel = MambaTPUKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)

    ref_out, ref_state = ref_kernel.forward_chunk(params, inputs, init_state, chunk_size=8)
    tpu_out, tpu_state = tpu_kernel.forward_chunk(params, inputs, init_state, chunk_size=8)

    np.testing.assert_allclose(ref_out, tpu_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ref_state.ssm, tpu_state.ssm, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _tpu_ready(), reason="TPU kernel requires TPU device availability")
@pytest.mark.parametrize("seq_len", [7, 15, 128, 256])
def test_tpu_kernel_various_seq_lengths(seq_len: int) -> None:
    """Test TPU kernel with various sequence lengths (tests padding alignment)."""
    batch, intermediate, state_size = 1, 2, 4

    key = jax.random.PRNGKey(seq_len)  # Use seq_len as seed for variety
    keys = jax.random.split(key, 7)

    hidden = jax.random.normal(keys[0], (batch, intermediate, seq_len))
    delta = jax.random.normal(keys[1], (batch, intermediate, seq_len))
    gate = jax.random.uniform(keys[2], (batch, intermediate, seq_len))
    B = jax.random.normal(keys[3], (batch, seq_len, state_size))
    C = jax.random.normal(keys[4], (batch, seq_len, state_size))

    params = MambaKernelParams(
        a_log=jax.random.normal(keys[5], (intermediate, state_size)),
        d=jax.random.uniform(keys[6], (intermediate,)),
    )
    inputs = MambaKernelInputs(hidden=hidden, delta=delta, B=B, C=C, gate=gate)
    init_state = MambaKernelState(ssm=jax.random.normal(keys[6], (batch, intermediate, state_size)))

    ref_kernel = MambaReferenceKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)
    tpu_kernel = MambaTPUKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)

    chunk_size = min(64, seq_len)
    ref_out, ref_state = ref_kernel.forward_chunk(params, inputs, init_state, chunk_size=chunk_size)
    tpu_out, tpu_state = tpu_kernel.forward_chunk(params, inputs, init_state, chunk_size=chunk_size)

    np.testing.assert_allclose(ref_out, tpu_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ref_state.ssm, tpu_state.ssm, rtol=1e-4, atol=1e-4)
