"""Delta rule kernels implemented with Pallas.

This file provides a correctness-first Pallas implementation of the recurrent
(delta rule) kernel used by DeltaNet.

Design goals:
- Always be safe on CPU-only machines: callers should catch failures and fall
  back to the reference JAX implementation.
- Match the reference math (float32 accumulation) for numerical stability.

Notes:
- This is not tuned for performance yet.
- Pallas lowering/availability can vary by JAX version and device; treat this
  as an optional acceleration path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def _pallas_is_usable() -> bool:
    try:
        from jax.experimental import pallas as pl  # noqa: F401
    except Exception:
        return False

    return any(device.platform == "gpu" for device in jax.devices())


def delta_rule_recurrent_pallas(
    q: jax.Array,  # [batch, heads, seq_len, key_dim]
    k: jax.Array,  # [batch, heads, seq_len, key_dim]
    v: jax.Array,  # [batch, heads, seq_len, value_dim]
    beta: jax.Array,  # [batch, heads, seq_len]
    initial_state: Optional[jax.Array] = None,  # [batch, heads, key_dim, value_dim]
    scale: Optional[float] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Pallas recurrent delta-rule kernel.

    Args:
        q: Query tensor [batch, heads, seq_len, key_dim]
        k: Key tensor [batch, heads, seq_len, key_dim]
        v: Value tensor [batch, heads, seq_len, value_dim]
        beta: Learning rate [batch, heads, seq_len]
        initial_state: Optional initial state [batch, heads, key_dim, value_dim]
        scale: Optional query scaling factor. Defaults to key_dim**-0.5.

    Returns:
        output: [batch, heads, seq_len, value_dim] (same dtype as q)
        final_state: [batch, heads, key_dim, value_dim] (float32)

    Raises:
        RuntimeError: If Pallas isn't available/usable on this machine.
    """

    if not _pallas_is_usable():
        raise RuntimeError("Pallas backend is not usable (need JAX Pallas + GPU).")

    from jax.experimental import pallas as pl

    if beta.ndim == 4:
        beta = beta.squeeze(-1)

    batch, heads, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]

    if scale is None:
        scale = key_dim**-0.5

    # Compute in float32 for stability (matches reference behavior).
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    beta_f32 = beta.astype(jnp.float32)

    if initial_state is None:
        state_in = jnp.zeros((batch, heads, key_dim, value_dim), dtype=jnp.float32)
    else:
        state_in = initial_state.astype(jnp.float32)

    out_shape = (
        jax.ShapeDtypeStruct((batch, heads, seq_len, value_dim), jnp.float32),
        jax.ShapeDtypeStruct((batch, heads, key_dim, value_dim), jnp.float32),
    )

    q_spec = pl.BlockSpec((1, 1, seq_len, key_dim), lambda b, h: (b, h, 0, 0))
    k_spec = pl.BlockSpec((1, 1, seq_len, key_dim), lambda b, h: (b, h, 0, 0))
    v_spec = pl.BlockSpec((1, 1, seq_len, value_dim), lambda b, h: (b, h, 0, 0))
    beta_spec = pl.BlockSpec((1, 1, seq_len), lambda b, h: (b, h, 0))
    state_spec = pl.BlockSpec((1, 1, key_dim, value_dim), lambda b, h: (b, h, 0, 0))

    out_spec = pl.BlockSpec((1, 1, seq_len, value_dim), lambda b, h: (b, h, 0, 0))
    state_out_spec = pl.BlockSpec(
        (1, 1, key_dim, value_dim), lambda b, h: (b, h, 0, 0)
    )

    scale_f32 = jnp.asarray(scale, dtype=jnp.float32)

    def kernel(q_ref, k_ref, v_ref, beta_ref, state_in_ref, out_ref, state_out_ref):
        S = pl.load(
            state_in_ref,
            (0, 0, pl.ds(0, key_dim), pl.ds(0, value_dim)),
        ).astype(jnp.float32)

        def body(t, S_carry):
            k_t = pl.load(k_ref, (0, 0, t, pl.ds(0, key_dim))).astype(jnp.float32)
            v_t = pl.load(v_ref, (0, 0, t, pl.ds(0, value_dim))).astype(jnp.float32)
            q_t = (
                pl.load(q_ref, (0, 0, t, pl.ds(0, key_dim))).astype(jnp.float32)
                * scale_f32
            )
            beta_t = pl.load(beta_ref, (0, 0, t)).astype(jnp.float32)

            v_old = jnp.einsum("kv,k->v", S_carry, k_t)
            v_delta = beta_t * (v_t - v_old)
            S_new = S_carry + jnp.einsum("k,v->kv", k_t, v_delta)
            o_t = jnp.einsum("k,kv->v", q_t, S_new)

            pl.store(out_ref, (0, 0, t, pl.ds(0, value_dim)), o_t)
            return S_new

        S = jax.lax.fori_loop(0, seq_len, body, S)
        pl.store(
            state_out_ref,
            (0, 0, pl.ds(0, key_dim), pl.ds(0, value_dim)),
            S,
        )

    out_f32, state_out_f32 = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=[q_spec, k_spec, v_spec, beta_spec, state_spec],
        out_specs=[out_spec, state_out_spec],
        grid=(batch, heads),
    )(q_f32, k_f32, v_f32, beta_f32, state_in)

    return out_f32.astype(q.dtype), state_out_f32
