"""Shared assertions for layer parity tests."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from linearnexus.kernels import KernelMode

Array = jax.Array


def _run_recurrent(
    layer,
    inputs: Array,
    *,
    attention_mask: Optional[Array] = None,
) -> Array:
    state = layer.init_state(inputs.shape[0], inputs.dtype)
    outputs = []
    for idx in range(inputs.shape[1]):
        token = inputs[:, idx : idx + 1, :]
        mask_token = None
        if attention_mask is not None:
            mask_token = attention_mask[:, idx : idx + 1]
        out, state = layer(
            token,
            state=state,
            mode=KernelMode.RECURRENT,
            chunk_size=1,
            attention_mask=mask_token,
        )
        outputs.append(out)
    return jnp.concatenate(outputs, axis=1)


def assert_chunk_recurrent_parity(
    layer,
    inputs: Array,
    *,
    chunk_size: int = 4,
    attention_mask: Optional[Array] = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Asserts that chunk and recurrent executions align numerically."""

    chunk_out, _ = layer(inputs, chunk_size=chunk_size, attention_mask=attention_mask)
    recurrent_out = _run_recurrent(layer, inputs, attention_mask=attention_mask)
    np.testing.assert_allclose(chunk_out, recurrent_out, rtol=rtol, atol=atol)


def assert_mask_behavior(
    layer,
    inputs: Array,
    mask: Array,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Asserts that masked tokens behave like explicit zeroed inputs."""

    masked_out, _ = layer(inputs, attention_mask=mask)
    zeros_out, _ = layer(inputs * 0.0)
    mask_expanded = (1.0 - mask)[..., None]
    np.testing.assert_allclose(
        masked_out * mask_expanded,
        zeros_out * mask_expanded,
        rtol=rtol,
        atol=atol,
    )
