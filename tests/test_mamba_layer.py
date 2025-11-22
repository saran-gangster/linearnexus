from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from linearnexus.kernels import PALLAS_AVAILABLE
from linearnexus.layers.mamba import MambaConfig, MambaLayer
from tests.helpers.parity import assert_chunk_recurrent_parity, assert_mask_behavior


def _pallas_ready() -> bool:
    return PALLAS_AVAILABLE and any(device.platform == "gpu" for device in jax.devices())


def test_chunk_and_recurrent_paths_align():
    rngs = nnx.Rngs(0)
    config = MambaConfig(hidden_size=32, intermediate_size=32, state_size=8, time_step_rank=8, conv_kernel=3)
    layer = MambaLayer(rngs, config)

    key = jax.random.PRNGKey(42)
    inputs = jax.random.normal(key, (2, 6, config.hidden_size))

    assert_chunk_recurrent_parity(layer, inputs, chunk_size=4)


def test_attention_mask_zeroes_out_tokens():
    rngs = nnx.Rngs(1)
    config = MambaConfig(hidden_size=16, intermediate_size=16, state_size=4, time_step_rank=4, conv_kernel=3)
    layer = MambaLayer(rngs, config)

    key = jax.random.PRNGKey(0)
    inputs = jax.random.normal(key, (1, 4, config.hidden_size))
    mask = jnp.array([[1.0, 1.0, 0.0, 0.0]], dtype=inputs.dtype)

    assert_mask_behavior(layer, inputs, mask)


@pytest.mark.skip(reason="Pallas backend disabled due to Triton limitations (Phase 0). Re-enable in Phase 1+ with handwritten Triton kernel.")
@pytest.mark.skipif(not _pallas_ready(), reason="Pallas backend requires a GPU + jax.experimental.pallas")
def test_pallas_backend_chunk_and_recurrent_paths_align():
    """Test Pallas backend kernel integration.
    
    **CURRENTLY DISABLED**: MambaPallasKernel raises NotImplementedError due to
    Triton/Pallas limitations. Sequential scan operations are not supported.
    """
    rngs = nnx.Rngs(4)
    config = MambaConfig(
        hidden_size=32,
        intermediate_size=32,
        state_size=8,
        time_step_rank=8,
        conv_kernel=3,
        kernel_backend="pallas",
    )
    layer = MambaLayer(rngs, config)

    key = jax.random.PRNGKey(123)
    inputs = jax.random.normal(key, (1, 10, config.hidden_size))

    assert_chunk_recurrent_parity(layer, inputs, chunk_size=5)
