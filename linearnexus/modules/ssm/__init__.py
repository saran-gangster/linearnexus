"""State-Space Model (SSM) blocks.

Implements selective state-space models for efficient sequence modeling:
- Mamba: Selective SSM with input-dependent dynamics (Mamba1)
- Mamba2: State Space Duality with multi-head structure
- RWKV: Linear attention with time decay (planned)
- S4: Structured state space sequences (planned)

SSMs offer O(n) complexity vs O(n²) for attention, with recurrent state
for efficient autoregressive generation.
"""

from .mamba import MambaBlock, MambaState
from .mamba2 import Mamba2Block, Mamba2State

__all__ = [
    "MambaBlock",
    "MambaState",
    "Mamba2Block",
    "Mamba2State",
]
