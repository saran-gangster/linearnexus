import flax.nnx as nnx
import pytest

from linearnexus.models import LMModel

# Import from the benchmark script (safe: guarded main)
from benchmark_architectures import get_standardized_configs


def _count_params(config) -> int:
    model = LMModel(config, rngs=nnx.Rngs(0))
    return int(model.count_params())


@pytest.mark.filterwarnings("ignore:.*\\.value.*:DeprecationWarning")
def test_benchmark_configs_are_param_matched():
    """Benchmark configs should have similar parameter counts.

    This test only checks model construction + parameter count (no training).
    """
    # Use smaller settings for test speed, but still exercise tuner.
    search_hidden_sizes = {
        # Wider candidates: some blocks need larger/smaller width to match.
        "mamba": [64, 96, 128, 160, 192],
        "mamba2": [64, 96, 128, 160, 192],
        "deltanet": [64, 96, 128, 160],
        "gated_deltanet": [64, 96, 128, 160, 192],
        "rwkv6": [64, 96, 128, 160],
        "rwkv7": [64, 96, 128, 160],
        "kda": [64, 96, 128, 160, 192],
    }

    configs = get_standardized_configs(
        vocab_size=65,
        n_layers=2,
        base_hidden_size=96,
        reference_arch="gpt",
        search_hidden_sizes=search_hidden_sizes,
    )

    param_counts = {name: _count_params(cfg) for name, cfg in configs.items()}

    min_params = min(param_counts.values())
    max_params = max(param_counts.values())

    # Allow some slack: architectures have different parameterizations.
    # Goal is "as close as possible"; this enforces the tuner isn't broken.
    rel_spread = (max_params - min_params) / max_params

    assert rel_spread <= 0.30, (
        f"Param spread too large: min={min_params}, max={max_params}, "
        f"rel_spread={rel_spread:.3f}. Full: {param_counts}"
    )
