"""Quick smoke test for the NNx Mamba layer."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.layers.mamba import MambaConfig, MambaLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--seq", type=int, default=16, help="Sequence length")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden size")
    parser.add_argument("--chunk", type=int, default=64, help="Chunk size override")
    parser.add_argument(
        "--kernel-backend",
        type=str,
        default="reference",
        choices=("reference", "pallas", "auto"),
        help="Kernel backend to use (reference, pallas, or auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rngs = nnx.Rngs(0)
    config = MambaConfig(
        hidden_size=args.hidden,
        intermediate_size=args.hidden,
        state_size=16,
        time_step_rank=16,
        conv_kernel=4,
        chunk_size=args.chunk,
        kernel_backend=args.kernel_backend,
    )
    layer = MambaLayer(rngs, config)
    inputs = jax.random.normal(jax.random.PRNGKey(0), (args.batch, args.seq, args.hidden))
    outputs, _ = layer(inputs)
    print("Output shape:", outputs.shape)
    print("Sample (first token):", jnp.asarray(outputs[0, 0, :5]))


if __name__ == "__main__":
    main()
