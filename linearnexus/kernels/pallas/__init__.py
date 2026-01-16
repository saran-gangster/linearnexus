"""Pallas-backed kernels.

These kernels are optional and should always be used behind a safe fallback to
reference JAX implementations.
"""

from .delta_rule import delta_rule_recurrent_pallas

__all__ = ["delta_rule_recurrent_pallas"]
