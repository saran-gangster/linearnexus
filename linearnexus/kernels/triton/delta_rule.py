"""Triton kernels for Delta Rule attention via JAX-Triton.

Currently provides a fused recurrent forward kernel with a reference backward
implementation to preserve correctness on non-GPU machines.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def _delta_rule_recurrent_reference(
    q: jax.Array,  # [batch, heads, seq, key_dim]
    k: jax.Array,  # [batch, heads, seq, key_dim]
    v: jax.Array,  # [batch, heads, seq, value_dim]
    beta: jax.Array,  # [batch, heads, seq]
    initial_state: Optional[jax.Array] = None,  # [batch, heads, key_dim, value_dim]
    scale: Optional[float] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Reference delta-rule recurrence for gradients.

    Args:
        q: [batch, heads, seq, key_dim]
        k: [batch, heads, seq, key_dim]
        v: [batch, heads, seq, value_dim]
        beta: [batch, heads, seq]
        initial_state: [batch, heads, key_dim, value_dim]
        scale: Optional scaling for q

    Returns:
        output: [batch, heads, seq, value_dim]
        final_state: [batch, heads, key_dim, value_dim]
    """
    batch, heads, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    orig_dtype = q.dtype

    if scale is None:
        scale = key_dim**-0.5

    if beta.ndim == 4:
        beta = beta.squeeze(-1)

    q = q.astype(jnp.float32) * jnp.asarray(scale, dtype=jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    if initial_state is None:
        S = jnp.zeros((batch, heads, key_dim, value_dim), dtype=jnp.float32)
    else:
        S = initial_state.astype(jnp.float32)

    def step(S, inputs):
        k_t, v_t, q_t, beta_t = inputs
        v_old = jnp.einsum("bhkv,bhk->bhv", S, k_t)
        v_delta = beta_t[..., None] * (v_t - v_old)
        S_new = S + jnp.einsum("bhk,bhv->bhkv", k_t, v_delta)
        o_t = jnp.einsum("bhk,bhkv->bhv", q_t, S_new)
        return S_new, o_t

    k_seq = jnp.transpose(k, (2, 0, 1, 3))
    v_seq = jnp.transpose(v, (2, 0, 1, 3))
    q_seq = jnp.transpose(q, (2, 0, 1, 3))
    beta_seq = jnp.transpose(beta, (2, 0, 1))

    final_state, outputs = jax.lax.scan(step, S, (k_seq, v_seq, q_seq, beta_seq))
    output = jnp.transpose(outputs, (1, 2, 0, 3))

    return output.astype(orig_dtype), final_state


def _triton_kernel_available() -> bool:
    try:
        import jax_triton  # type: ignore[import-not-found]  # noqa: F401
        import triton  # type: ignore[import-not-found]  # noqa: F401
        import triton.language as tl  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return False
    return any(device.platform == "gpu" for device in jax.devices())


def _delta_rule_recurrent_triton_impl(
    q: jax.Array,  # [batch, heads, seq, key_dim]
    k: jax.Array,  # [batch, heads, seq, key_dim]
    v: jax.Array,  # [batch, heads, seq, value_dim]
    beta: jax.Array,  # [batch, heads, seq]
    initial_state: Optional[jax.Array],
    scale: float,
) -> Tuple[jax.Array, jax.Array]:
    import jax_triton as jt  # type: ignore[import-not-found]
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    batch, heads, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    if beta.ndim == 4:
        beta = beta.squeeze(-1)

    if key_dim <= 0 or value_dim <= 0:
        raise ValueError("Key/value dimensions must be positive.")

    bk = triton.next_power_of_2(key_dim)
    bv = min(triton.next_power_of_2(value_dim), 32)
    if key_dim > bk:
        raise ValueError("Key dim must be <= BK for Triton kernel.")

    # Layout: [B*T, H, K] for Q/K and [B*T, H, V] for V and output
    q_bt = jnp.transpose(q, (0, 2, 1, 3)).reshape(batch * seq_len, heads, key_dim)
    k_bt = jnp.transpose(k, (0, 2, 1, 3)).reshape(batch * seq_len, heads, key_dim)
    v_bt = jnp.transpose(v, (0, 2, 1, 3)).reshape(batch * seq_len, heads, value_dim)
    beta_bt = jnp.transpose(beta, (0, 2, 1)).reshape(batch * seq_len, heads)

    if initial_state is None:
        h0 = jnp.zeros((batch, heads, key_dim, value_dim), dtype=jnp.float32)
        use_initial_state = False
    else:
        h0 = initial_state.astype(jnp.float32)
        use_initial_state = True

    out_shape = (
        jax.ShapeDtypeStruct((batch * seq_len, heads, value_dim), q.dtype),
        jax.ShapeDtypeStruct((batch, heads, key_dim, value_dim), jnp.float32),
    )

    @triton.jit
    def _delta_rule_recurrent_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        beta_ptr,
        h0_ptr,
        scale_ptr,
        o_ptr,
        ht_ptr,
        T,
        H,
        K,
        V,
        BK: tl.constexpr,
        BV: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_nh = tl.program_id(2)
        i_n = i_nh // H
        i_h = i_nh % H
        bos = i_n * T

        p_q = q_ptr + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
        p_k = k_ptr + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
        p_v = v_ptr + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        p_o = o_ptr + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        p_beta = beta_ptr + bos * H + i_h

        mask_k = (i_k * BK + tl.arange(0, BK)) < K
        mask_v = (i_v * BV + tl.arange(0, BV)) < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        b_h = tl.zeros([BV, BK], dtype=tl.float32)
        if USE_INITIAL_STATE:
            p_h0 = h0_ptr + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (
                i_v * BV + tl.arange(0, BV)[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        scale_val = tl.load(scale_ptr).to(tl.float32)

        for _ in range(0, T):
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale_val
            b_v_minus = tl.sum(b_h * b_k[None, :], axis=1)
            b_v = b_v - b_v_minus
            b_beta = tl.load(p_beta).to(tl.float32)
            b_v = b_v * b_beta
            b_h += b_k[None, :] * b_v[:, None]
            b_o = tl.sum(b_h * b_q[None, :], axis=1)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            p_q += H * K
            p_k += H * K
            p_v += H * V
            p_o += H * V
            p_beta += H

        p_ht = ht_ptr + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (
            i_v * BV + tl.arange(0, BV)[:, None]
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

    scale_arr = jnp.asarray(scale, dtype=jnp.float32)

    o_bt, final_state = jt.triton_call(
        q_bt,
        k_bt,
        v_bt,
        beta_bt,
        h0,
        scale_arr,
        kernel=_delta_rule_recurrent_fwd_kernel,
        out_shape=out_shape,
        grid=(triton.cdiv(value_dim, bv), 1, batch * heads),
        T=seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        BK=bk,
        BV=bv,
        USE_INITIAL_STATE=use_initial_state,
        num_warps=1,
        num_stages=1,
    )

    output = o_bt.reshape(batch, seq_len, heads, value_dim).transpose(0, 2, 1, 3)
    return output, final_state


@jax.custom_vjp
def delta_rule_recurrent_triton(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta: jax.Array,
    initial_state: Optional[jax.Array] = None,
    scale: Optional[float] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Triton-accelerated recurrent delta rule (forward) with reference backward.

    Falls back to the reference implementation if Triton is unavailable.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if not _triton_kernel_available():
        return _delta_rule_recurrent_reference(q, k, v, beta, initial_state, scale)

    return _delta_rule_recurrent_triton_impl(q, k, v, beta, initial_state, scale)


def _delta_rule_recurrent_triton_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta: jax.Array,
    initial_state: Optional[jax.Array],
    scale: Optional[float],
):
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if not _triton_kernel_available():
        y, final_state = _delta_rule_recurrent_reference(q, k, v, beta, initial_state, scale)
    else:
        y, final_state = _delta_rule_recurrent_triton_impl(q, k, v, beta, initial_state, scale)

    return (y, final_state), (q, k, v, beta, initial_state, scale)


def _delta_rule_recurrent_triton_bwd(res, g):
    q, k, v, beta, initial_state, scale = res
    grad_out, grad_state = g

    if grad_state is None:
        batch, heads, _, key_dim = q.shape
        value_dim = v.shape[-1]
        grad_state = jnp.zeros((batch, heads, key_dim, value_dim), dtype=jnp.float32)

    def ref_fn(q, k, v, beta, initial_state, scale):
        return _delta_rule_recurrent_reference(q, k, v, beta, initial_state, scale)

    _, vjp_fn = jax.vjp(ref_fn, q, k, v, beta, initial_state, scale)
    dq, dk, dv, dbeta, dstate, _ = vjp_fn((grad_out, grad_state))

    return dq, dk, dv, dbeta, dstate, None


delta_rule_recurrent_triton.defvjp(
    _delta_rule_recurrent_triton_fwd,
    _delta_rule_recurrent_triton_bwd,
)
