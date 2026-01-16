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
        import jax.experimental.pallas.mosaic_gpu as plgpu  # noqa: F401
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
    import jax.experimental.pallas.mosaic_gpu as plgpu

    if beta.ndim == 4:
        beta = beta.squeeze(-1)

    batch, heads, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]

    if scale is None:
        scale = key_dim**-0.5

    # Compute in float32 for stability (matches reference behavior).
    # IMPORTANT: Mosaic kernels can't close over traced JAX values (e.g. a
    # jnp.asarray(scale) captured in the kernel body). Pre-apply scaling here.
    q_f32 = q.astype(jnp.float32) * jnp.asarray(scale, dtype=jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    beta_f32 = beta.astype(jnp.float32)

    if initial_state is None:
        state_in = jnp.zeros((batch, heads, key_dim, value_dim), dtype=jnp.float32)
    else:
        state_in = initial_state.astype(jnp.float32)

    # Mosaic GPU lowering in some JAX versions cannot lower generic masked
    # loads/stores emitted by `pl.load` / `pl.store`. Follow the FlashAttention3
    # reference style (docs/reference/attention_mgpu.py): move all global memory
    # traffic through explicit TMA copies to SMEM, then compute from SMEM.
    #
    # For now, keep the supported shapes conservative to ensure TMA-friendly
    # copies (and let callers fall back to the reference kernel otherwise).
    if key_dim % 8 != 0 or value_dim % 8 != 0:
        raise RuntimeError(
            "Pallas delta-rule currently requires key_dim and value_dim to be multiples of 8 "
            "(TMA/SMEM copy friendly)."
        )

    out_shape = [
        jax.ShapeDtypeStruct((batch, heads, seq_len, value_dim), q.dtype),
        jax.ShapeDtypeStruct((batch, heads, key_dim, value_dim), jnp.float32),
    ]

    def kernel(q_ref, k_ref, v_ref, beta_ref, state_in_ref, out_ref, state_out_ref, scoped):
        b = jax.lax.axis_index("b")
        h = jax.lax.axis_index("h")
        (
            q_smem,
            k_smem,
            v_smem,
            out_smem,
            state_smem,
            v_old_smem,
            o_base_smem,
            v_delta_smem,
            qk_smem,
        ), (
            q_barrier,
            k_barrier,
            v_barrier,
            out_barrier,
            state_barrier,
        ) = scoped

        # Load initial state into SMEM once.
        plgpu.copy_gmem_to_smem(
            state_in_ref.at[b, h, pl.ds(0, key_dim), pl.ds(0, value_dim)],
            state_smem,
            state_barrier.at[0],
        )
        plgpu.barrier_wait(state_barrier.at[0])

        def body(t, _):
            plgpu.copy_gmem_to_smem(
                q_ref.at[b, h, t, pl.ds(0, key_dim)],
                q_smem,
                q_barrier.at[0],
            )
            plgpu.copy_gmem_to_smem(
                k_ref.at[b, h, t, pl.ds(0, key_dim)],
                k_smem,
                k_barrier.at[0],
            )
            plgpu.copy_gmem_to_smem(
                v_ref.at[b, h, t, pl.ds(0, value_dim)],
                v_smem,
                v_barrier.at[0],
            )

            plgpu.barrier_wait(q_barrier.at[0])
            plgpu.barrier_wait(k_barrier.at[0])
            plgpu.barrier_wait(v_barrier.at[0])

            # Mosaic GPU lowering (JAX 0.8.2) currently rejects general
            # broadcasting primitives. Implement delta-rule with explicit
            # scalar loops over fixed sizes.
            # NOTE: Mosaic GPU TMA copies must be multiples of 128 bytes.
            # Loading beta via TMA (4 bytes) fails. Instead, use a scalar load.
            beta_t = pl.load(beta_ref, (b, h, t)).astype(jnp.float32)

            # Zero accumulators without broadcasting.
            qk_smem.at[0][...] = jnp.array(0.0, dtype=jnp.float32)

            @pl.loop(0, value_dim)
            def _zero_vecs(j):
                v_old_smem.at[j][...] = jnp.array(0.0, dtype=jnp.float32)
                o_base_smem.at[j][...] = jnp.array(0.0, dtype=jnp.float32)

            # Accumulate:
            #   v_old[j]  = sum_i state[i, j] * k[i]
            #   o_base[j] = sum_i state[i, j] * q[i]
            #   qk        = sum_i q[i] * k[i]
            @pl.loop(0, key_dim)
            def _accum_i(i):
                k_i = k_smem.at[i][...]
                q_i = q_smem.at[i][...]
                qk_smem.at[0][...] = qk_smem.at[0][...] + (q_i * k_i)

                @pl.loop(0, value_dim)
                def _accum_j(j):
                    s_ij = state_smem.at[i, j][...]
                    v_old_smem.at[j][...] = v_old_smem.at[j][...] + (s_ij * k_i)
                    o_base_smem.at[j][...] = o_base_smem.at[j][...] + (s_ij * q_i)

            # v_delta[j] = beta * (v[j] - v_old[j])
            @pl.loop(0, value_dim)
            def _compute_v_delta(j):
                v_j = v_smem.at[j][...]
                v_delta_smem.at[j][...] = beta_t * (v_j - v_old_smem.at[j][...])

            # out[j] = o_base[j] + qk * v_delta[j]
            @pl.loop(0, value_dim)
            def _write_out(j):
                out_val = o_base_smem.at[j][...] + (qk_smem.at[0][...] * v_delta_smem.at[j][...])
                out_smem.at[j][...] = out_val.astype(out_ref.dtype)

            # state[i, j] += k[i] * v_delta[j]
            @pl.loop(0, key_dim)
            def _update_i(i):
                k_i = k_smem.at[i][...]

                @pl.loop(0, value_dim)
                def _update_j(j):
                    state_smem.at[i, j][...] = state_smem.at[i, j][...] + (k_i * v_delta_smem.at[j][...])

            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(
                out_smem,
                out_ref.at[b, h, t, pl.ds(0, value_dim)],
            )
            plgpu.wait_smem_to_gmem(0)

            return None

        jax.lax.fori_loop(0, seq_len, body, None)

        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(
            state_smem,
            state_out_ref.at[b, h, pl.ds(0, key_dim), pl.ds(0, value_dim)],
        )
        plgpu.wait_smem_to_gmem(0)

    def entry(q_ref, k_ref, v_ref, beta_ref, state_in_ref, out_ref, state_out_ref):
        q_smem = plgpu.SMEM((key_dim,), jnp.float32)
        k_smem = plgpu.SMEM((key_dim,), jnp.float32)
        v_smem = plgpu.SMEM((value_dim,), jnp.float32)
        out_smem = plgpu.SMEM((value_dim,), q.dtype)
        state_smem = plgpu.SMEM((key_dim, value_dim), jnp.float32)
        v_old_smem = plgpu.SMEM((value_dim,), jnp.float32)
        o_base_smem = plgpu.SMEM((value_dim,), jnp.float32)
        v_delta_smem = plgpu.SMEM((value_dim,), jnp.float32)
        qk_smem = plgpu.SMEM((1,), jnp.float32)
        pl.run_scoped(
            lambda *scoped: kernel(q_ref, k_ref, v_ref, beta_ref, state_in_ref, out_ref, state_out_ref, scoped),
            (q_smem, k_smem, v_smem, out_smem, state_smem, v_old_smem, o_base_smem, v_delta_smem, qk_smem),
            (
                plgpu.Barrier(num_barriers=1),
                plgpu.Barrier(num_barriers=1),
                plgpu.Barrier(num_barriers=1),
                plgpu.Barrier(num_barriers=1),
                plgpu.Barrier(num_barriers=1),
            ),
            collective_axes="wg",
        )

    compiled = plgpu.kernel(
        entry,
        out_shape=out_shape,
        grid=(batch, heads),
        grid_names=("b", "h"),
        # Correctness-first: single warpgroup thread.
        num_threads=1,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )

    out, state_out = compiled(q_f32, k_f32, v_f32, beta_f32, state_in)
    return out, state_out
