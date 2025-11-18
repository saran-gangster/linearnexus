"""NNx-based Mamba layer wired to the reference kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.core import ConfigBase, ConvState, RecurrentState, depthwise_conv1d_causal, select_mode
from linearnexus.kernels import (
    KernelMode,
    MambaPallasKernel,
    MambaKernelInputs,
    MambaKernelParams,
    MambaKernelState,
    MambaReferenceKernel,
    PALLAS_AVAILABLE,
)


_ACT_FNS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "identity": lambda x: x,
}


@dataclass
class MambaConfig(ConfigBase):
    """Configuration container for the Mamba layer."""

    hidden_size: int = 256
    state_size: int = 16
    conv_kernel: int = 4
    use_conv_bias: bool = True
    intermediate_size: int = 256
    time_step_rank: int = 128
    use_bias: bool = True
    hidden_act: str = "silu"
    chunk_size: int = 64
    kernel_backend: str = "reference"


@dataclass
class MambaLayerState:
    """State tracked between invocations for caching / decoding."""

    conv_state: ConvState
    ssm_state: RecurrentState
    position: jnp.int32

    @classmethod
    def initialize(
        cls,
        *,
        batch_size: int,
        conv_kernel: int,
        intermediate_size: int,
        state_size: int,
        dtype: jnp.dtype,
    ) -> "MambaLayerState":
        return cls(
            conv_state=ConvState.zeros(
                batch_size=batch_size,
                kernel_size=conv_kernel,
                channels=intermediate_size,
                dtype=dtype,
            ),
            ssm_state=RecurrentState.zeros(
                batch_size=batch_size,
                channels=intermediate_size,
                state_size=state_size,
                dtype=dtype,
            ),
            position=jnp.array(0, dtype=jnp.int32),
        )


def _activation_fn(name: str) -> Callable[[jax.Array], jax.Array]:
    if name not in _ACT_FNS:
        raise ValueError(f"Unsupported activation '{name}'")
    return _ACT_FNS[name]


def _init_conv_weight(rng: jax.Array, kernel: int, channels: int) -> jax.Array:
    scale = 1.0 / math.sqrt(kernel * channels)
    return jax.random.normal(rng, (kernel, channels)) * scale


def _has_gpu_support() -> bool:
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except RuntimeError:  # pragma: no cover - triggered when JAX has no default backend
        return False


class MambaLayer(nnx.Module):
    """NNx Mamba layer with reference selective-scan kernel."""

    def __init__(self, rngs: nnx.Rngs, config: MambaConfig):
        self.config = config
        act_fn = _activation_fn(config.hidden_act)
        self.activation: Callable[[jax.Array], jax.Array] = act_fn

        self.in_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size * 2,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.x_proj = nnx.Linear(
            config.intermediate_size,
            config.time_step_rank + config.state_size * 2,
            use_bias=False,
            rngs=rngs,
        )
        self.dt_proj = nnx.Linear(
            config.time_step_rank,
            config.intermediate_size,
            use_bias=True,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        conv_rng = rngs.params()
        self.conv_weight = nnx.Param(_init_conv_weight(conv_rng, config.conv_kernel, config.intermediate_size))
        self.conv_bias = nnx.Param(jnp.zeros((config.intermediate_size,))) if config.use_conv_bias else None

        a_base = jnp.arange(1, config.state_size + 1, dtype=jnp.float32)
        a = jnp.tile(a_base[None, :], (config.intermediate_size, 1))
        self.a_log = nnx.Param(jnp.log(a))
        self.d = nnx.Param(jnp.ones((config.intermediate_size,), dtype=jnp.float32))

        self.kernel = self._build_kernel(config.kernel_backend)

    def _build_kernel(self, backend: str):
        backend_normalized = (backend or "reference").lower()
        if backend_normalized == "auto":
            backend_normalized = "pallas" if (_has_gpu_support() and PALLAS_AVAILABLE) else "reference"

        if backend_normalized == "pallas":
            if not PALLAS_AVAILABLE:
                raise ImportError("Pallas backend requested but jax.experimental.pallas is not available")
            return MambaPallasKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)
        if backend_normalized == "reference":
            return MambaReferenceKernel(mode=KernelMode.CHUNK, dtype=jnp.float32)
        raise ValueError(
            "Unsupported kernel backend '{backend}'. Choose from {'reference', 'pallas', 'auto'}.".format(
                backend=backend
            )
        )

    def init_state(self, batch_size: int, dtype: jnp.dtype) -> MambaLayerState:
        return MambaLayerState.initialize(
            batch_size=batch_size,
            conv_kernel=self.config.conv_kernel,
            intermediate_size=self.config.intermediate_size,
            state_size=self.config.state_size,
            dtype=dtype,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        state: Optional[MambaLayerState] = None,
        attention_mask: Optional[jax.Array] = None,
        mode: Optional[KernelMode] = None,
        chunk_size: Optional[int] = None,
    ) -> tuple[jax.Array, MambaLayerState]:
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        chunk_size = chunk_size or self.config.chunk_size
        if state is None:
            state = self.init_state(batch_size, dtype)
        if mode is None:
            mode = select_mode(seq_len, threshold=chunk_size)

        projected = self.in_proj(hidden_states)
        hidden, gate = jnp.split(projected, 2, axis=-1)

        if attention_mask is not None:
            mask = attention_mask[..., None]
            hidden = hidden * mask
            gate = gate * mask

        conv_out, conv_cache = depthwise_conv1d_causal(
            hidden,
            self.conv_weight.value,
            self.conv_bias.value if self.conv_bias is not None else None,
            cache=state.conv_state.buffer,
        )
        conv_out = self.activation(conv_out)

        if attention_mask is not None:
            conv_out = conv_out * attention_mask[..., None]

        x_proj_out = self.x_proj(conv_out)
        time_step, B, C = jnp.split(
            x_proj_out,
            [self.config.time_step_rank, self.config.time_step_rank + self.config.state_size],
            axis=-1,
        )
        delta = jax.nn.softplus(self.dt_proj(time_step))

        hidden_kernel = jnp.swapaxes(conv_out, 1, 2)
        gate_kernel = jnp.swapaxes(self.activation(gate), 1, 2)
        delta_kernel = jnp.swapaxes(delta, 1, 2)

        kernel_inputs = MambaKernelInputs(
            hidden=hidden_kernel,
            delta=delta_kernel,
            B=B,
            C=C,
            gate=gate_kernel,
        )
        kernel_params = MambaKernelParams(
            a_log=self.a_log.value.astype(dtype),
            d=self.d.value.astype(dtype),
        )
        kernel_state = MambaKernelState(ssm=state.ssm_state.value)

        if mode == KernelMode.CHUNK:
            kernel_outputs, new_kernel_state = self.kernel.forward_chunk(
                kernel_params,
                kernel_inputs,
                kernel_state,
                chunk_size=chunk_size,
            )
        else:
            kernel_outputs, new_kernel_state = self.kernel.forward_recurrent(
                kernel_params,
                kernel_inputs,
                kernel_state,
            )

        kernel_outputs = kernel_outputs.transpose(0, 2, 1)
        contextualized = self.out_proj(kernel_outputs)

        new_state = MambaLayerState(
            conv_state=state.conv_state.update(conv_cache),
            ssm_state=state.ssm_state.update(new_kernel_state.ssm),
            position=state.position + seq_len,
        )
        return contextualized, new_state
