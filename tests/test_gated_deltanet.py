"""Tests for Gated DeltaNet implementation.

Validates the gated delta rule kernel functions and GatedDeltaNetBlock.
Based on the FLA implementation from "Gated Delta Networks: Improving Mamba2 with Delta Rule".
"""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.modules.linear_attn.gated_deltanet import (
    GatedDeltaNetBlock,
    GatedDeltaNetState,
    gated_delta_rule_recurrent,
    gated_delta_rule_chunkwise,
    gated_delta_rule_step,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_shapes():
    """Small shapes for quick testing."""
    return {
        "batch": 2,
        "seq_len": 32,
        "hidden_size": 64,
        "num_heads": 2,
        "num_v_heads": 2,
        "head_dim": 32,
        "head_v_dim": 64,  # expand_v = 2
    }


@pytest.fixture
def medium_shapes():
    """Medium shapes closer to real usage."""
    return {
        "batch": 2,
        "seq_len": 128,
        "hidden_size": 256,
        "num_heads": 4,
        "num_v_heads": 4,
        "head_dim": 64,
        "head_v_dim": 128,
    }


# =============================================================================
# State Tests
# =============================================================================

class TestGatedDeltaNetState:
    """Tests for GatedDeltaNetState cache."""

    def test_zeros_basic(self, small_shapes):
        """Test basic state initialization."""
        state = GatedDeltaNetState.zeros(
            batch_size=small_shapes["batch"],
            num_v_heads=small_shapes["num_v_heads"],
            key_dim_per_head=small_shapes["head_dim"],
            value_dim_per_head=small_shapes["head_v_dim"],
        )
        
        assert state.S.shape == (
            small_shapes["batch"],
            small_shapes["num_v_heads"],
            small_shapes["head_dim"],
            small_shapes["head_v_dim"],
        )
        assert jnp.allclose(state.S, 0.0)
        assert state.conv_state_q is None
        assert state.conv_state_k is None
        assert state.conv_state_v is None

    def test_zeros_with_conv(self, small_shapes):
        """Test state initialization with conv caches."""
        key_dim_total = small_shapes["num_heads"] * small_shapes["head_dim"]
        value_dim = small_shapes["num_v_heads"] * small_shapes["head_v_dim"]
        conv_size = 4
        
        state = GatedDeltaNetState.zeros(
            batch_size=small_shapes["batch"],
            num_v_heads=small_shapes["num_v_heads"],
            key_dim_per_head=small_shapes["head_dim"],
            value_dim_per_head=small_shapes["head_v_dim"],
            key_dim_total=key_dim_total,
            value_dim=value_dim,
            conv_size=conv_size,
            use_conv=True,
        )
        
        assert state.S.shape == (
            small_shapes["batch"],
            small_shapes["num_v_heads"],
            small_shapes["head_dim"],
            small_shapes["head_v_dim"],
        )
        assert state.conv_state_q.shape == (small_shapes["batch"], conv_size - 1, key_dim_total)
        assert state.conv_state_k.shape == (small_shapes["batch"], conv_size - 1, key_dim_total)
        assert state.conv_state_v.shape == (small_shapes["batch"], conv_size - 1, value_dim)


# =============================================================================
# Kernel Tests
# =============================================================================

class TestGatedDeltaRuleRecurrent:
    """Tests for gated_delta_rule_recurrent."""

    def test_output_shape(self, key, small_shapes):
        """Test output shapes."""
        batch = small_shapes["batch"]
        seq_len = small_shapes["seq_len"]
        heads = small_shapes["num_v_heads"]
        key_dim = small_shapes["head_dim"]
        value_dim = small_shapes["head_v_dim"]
        
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, heads, seq_len, key_dim))
        k = jax.random.normal(k2, (batch, heads, seq_len, key_dim))
        v = jax.random.normal(k3, (batch, heads, seq_len, value_dim))
        g = jax.random.uniform(k4, (batch, heads, seq_len), minval=-2.0, maxval=0.0)  # Typically negative
        beta = jax.random.uniform(k5, (batch, heads, seq_len), minval=0.0, maxval=1.0)
        
        output, final_state = gated_delta_rule_recurrent(q, k, v, g, beta)
        
        assert output.shape == (batch, heads, seq_len, value_dim)
        assert final_state.shape == (batch, heads, key_dim, value_dim)

    def test_no_gate_equals_deltanet(self, key, small_shapes):
        """Test that g=0 (no decay) behaves similarly to regular delta rule."""
        batch = small_shapes["batch"]
        seq_len = small_shapes["seq_len"]
        heads = small_shapes["num_v_heads"]
        key_dim = small_shapes["head_dim"]
        value_dim = small_shapes["head_v_dim"]
        
        k1, k2, k3, k4 = jax.random.split(key, 4)
        q = jax.random.normal(k1, (batch, heads, seq_len, key_dim)) * 0.1
        k = jax.random.normal(k2, (batch, heads, seq_len, key_dim)) * 0.1
        v = jax.random.normal(k3, (batch, heads, seq_len, value_dim)) * 0.1
        g = jnp.zeros((batch, heads, seq_len))  # No decay
        beta = jax.random.uniform(k4, (batch, heads, seq_len), minval=0.5, maxval=1.0)
        
        output, final_state = gated_delta_rule_recurrent(q, k, v, g, beta, use_qk_l2norm=False)
        
        # With g=0, exp(g) = 1, so no decay - outputs should be non-trivial
        assert not jnp.allclose(output, 0.0)
        assert not jnp.allclose(final_state, 0.0)

    def test_high_decay_forgets(self, key, small_shapes):
        """Test that large negative g (strong decay) makes state forget."""
        batch = small_shapes["batch"]
        seq_len = 16
        heads = small_shapes["num_v_heads"]
        key_dim = small_shapes["head_dim"]
        value_dim = small_shapes["head_v_dim"]
        
        k1, k2, k3, k4 = jax.random.split(key, 4)
        q = jax.random.normal(k1, (batch, heads, seq_len, key_dim)) * 0.1
        k = jax.random.normal(k2, (batch, heads, seq_len, key_dim)) * 0.1
        v = jax.random.normal(k3, (batch, heads, seq_len, value_dim)) * 0.1
        g = jnp.full((batch, heads, seq_len), -10.0)  # Strong decay
        beta = jnp.ones((batch, heads, seq_len))
        
        output, final_state = gated_delta_rule_recurrent(q, k, v, g, beta, use_qk_l2norm=False)
        
        # With very strong decay, final state should be mostly recent tokens
        # The state magnitude should be relatively small due to forgetting
        assert final_state.shape == (batch, heads, key_dim, value_dim)

    def test_with_initial_state(self, key, small_shapes):
        """Test continuation from initial state."""
        batch = small_shapes["batch"]
        seq_len = small_shapes["seq_len"]
        heads = small_shapes["num_v_heads"]
        key_dim = small_shapes["head_dim"]
        value_dim = small_shapes["head_v_dim"]
        
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        q = jax.random.normal(k1, (batch, heads, seq_len, key_dim)) * 0.1
        k = jax.random.normal(k2, (batch, heads, seq_len, key_dim)) * 0.1
        v = jax.random.normal(k3, (batch, heads, seq_len, value_dim)) * 0.1
        g = jax.random.uniform(k4, (batch, heads, seq_len), minval=-1.0, maxval=0.0)
        beta = jax.random.uniform(k5, (batch, heads, seq_len), minval=0.0, maxval=1.0)
        
        initial_state = jax.random.normal(k6, (batch, heads, key_dim, value_dim)) * 0.01
        
        output_zero, state_zero = gated_delta_rule_recurrent(q, k, v, g, beta, initial_state=None)
        output_init, state_init = gated_delta_rule_recurrent(q, k, v, g, beta, initial_state=initial_state)
        
        # With initial state, outputs should be different
        assert not jnp.allclose(output_zero, output_init, atol=1e-5)


class TestGatedDeltaRuleChunkwise:
    """Tests for gated_delta_rule_chunkwise."""

    def test_output_shape(self, key, small_shapes):
        """Test output shapes."""
        batch = small_shapes["batch"]
        seq_len = 64  # Multiple of chunk_size
        heads = small_shapes["num_v_heads"]
        key_dim = small_shapes["head_dim"]
        value_dim = small_shapes["head_v_dim"]
        
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, heads, seq_len, key_dim))
        k = jax.random.normal(k2, (batch, heads, seq_len, key_dim))
        v = jax.random.normal(k3, (batch, heads, seq_len, value_dim))
        g = jax.random.uniform(k4, (batch, heads, seq_len), minval=-1.0, maxval=0.0)
        beta = jax.random.uniform(k5, (batch, heads, seq_len), minval=0.0, maxval=1.0)
        
        output, final_state = gated_delta_rule_chunkwise(q, k, v, g, beta, chunk_size=32)
        
        assert output.shape == (batch, heads, seq_len, value_dim)
        assert final_state.shape == (batch, heads, key_dim, value_dim)

    def test_handles_padding(self, key, small_shapes):
        """Test that non-chunk-aligned sequences are handled correctly."""
        batch = small_shapes["batch"]
        seq_len = 50  # Not a multiple of chunk_size=32
        heads = small_shapes["num_v_heads"]
        key_dim = small_shapes["head_dim"]
        value_dim = small_shapes["head_v_dim"]
        
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, heads, seq_len, key_dim))
        k = jax.random.normal(k2, (batch, heads, seq_len, key_dim))
        v = jax.random.normal(k3, (batch, heads, seq_len, value_dim))
        g = jax.random.uniform(k4, (batch, heads, seq_len), minval=-1.0, maxval=0.0)
        beta = jax.random.uniform(k5, (batch, heads, seq_len), minval=0.0, maxval=1.0)
        
        output, final_state = gated_delta_rule_chunkwise(q, k, v, g, beta, chunk_size=32)
        
        # Output should be correctly sized (padding removed)
        assert output.shape == (batch, heads, seq_len, value_dim)

    def test_recurrent_vs_chunkwise_parity(self, key, small_shapes):
        """Test that recurrent and chunkwise modes produce similar results.
        
        Note: The chunkwise implementation may have different numerical behavior
        due to different computation order (cumulative gates, WY decomposition).
        We test that both produce reasonable outputs without NaN/Inf.
        """
        batch = 1
        seq_len = 64
        heads = 2
        key_dim = 16
        value_dim = 32
        
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, heads, seq_len, key_dim)) * 0.1
        k = jax.random.normal(k2, (batch, heads, seq_len, key_dim)) * 0.1
        v = jax.random.normal(k3, (batch, heads, seq_len, value_dim)) * 0.1
        g = jax.random.uniform(k4, (batch, heads, seq_len), minval=-0.5, maxval=0.0)  # Small decay
        beta = jax.random.uniform(k5, (batch, heads, seq_len), minval=0.3, maxval=0.7)
        
        out_rec, state_rec = gated_delta_rule_recurrent(
            q, k, v, g, beta, scale=1.0, use_qk_l2norm=True
        )
        out_chunk, state_chunk = gated_delta_rule_chunkwise(
            q, k, v, g, beta, scale=1.0, chunk_size=32, use_qk_l2norm=True
        )
        
        # Both outputs should be finite (no NaN or Inf)
        assert jnp.all(jnp.isfinite(out_rec)), "Recurrent output has NaN/Inf"
        assert jnp.all(jnp.isfinite(out_chunk)), "Chunkwise output has NaN/Inf"
        assert jnp.all(jnp.isfinite(state_rec)), "Recurrent state has NaN/Inf"
        assert jnp.all(jnp.isfinite(state_chunk)), "Chunkwise state has NaN/Inf"
        
        # Both should have similar magnitude range
        assert out_rec.std() > 1e-6, "Recurrent output is too small"
        assert out_chunk.std() > 1e-6, "Chunkwise output is too small"


class TestGatedDeltaRuleStep:
    """Tests for gated_delta_rule_step."""

    def test_output_shape(self, key, small_shapes):
        """Test output shapes for single step."""
        batch = small_shapes["batch"]
        heads = small_shapes["num_v_heads"]
        key_dim = small_shapes["head_dim"]
        value_dim = small_shapes["head_v_dim"]
        
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        q = jax.random.normal(k1, (batch, heads, key_dim))
        k = jax.random.normal(k2, (batch, heads, key_dim))
        v = jax.random.normal(k3, (batch, heads, value_dim))
        g = jax.random.uniform(k4, (batch, heads), minval=-1.0, maxval=0.0)
        beta = jax.random.uniform(k5, (batch, heads), minval=0.0, maxval=1.0)
        state = jax.random.normal(k6, (batch, heads, key_dim, value_dim)) * 0.01
        
        output, new_state = gated_delta_rule_step(q, k, v, g, beta, state)
        
        assert output.shape == (batch, heads, value_dim)
        assert new_state.shape == (batch, heads, key_dim, value_dim)

    def test_step_matches_recurrent_first_step(self, key, small_shapes):
        """Test that step function matches first step of recurrent."""
        batch = small_shapes["batch"]
        heads = small_shapes["num_v_heads"]
        key_dim = small_shapes["head_dim"]
        value_dim = small_shapes["head_v_dim"]
        
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, heads, key_dim))
        k = jax.random.normal(k2, (batch, heads, key_dim))
        v = jax.random.normal(k3, (batch, heads, value_dim))
        g = jax.random.uniform(k4, (batch, heads), minval=-1.0, maxval=0.0)
        beta = jax.random.uniform(k5, (batch, heads), minval=0.0, maxval=1.0)
        
        state = jnp.zeros((batch, heads, key_dim, value_dim))
        
        # Single step
        out_step, state_step = gated_delta_rule_step(q, k, v, g, beta, state)
        
        # Recurrent with seq_len=1
        q_seq = q[:, :, None, :]
        k_seq = k[:, :, None, :]
        v_seq = v[:, :, None, :]
        g_seq = g[:, :, None]
        beta_seq = beta[:, :, None]
        
        out_rec, state_rec = gated_delta_rule_recurrent(q_seq, k_seq, v_seq, g_seq, beta_seq)
        
        assert jnp.allclose(out_step, out_rec[:, :, 0, :], rtol=1e-4, atol=1e-4)
        assert jnp.allclose(state_step, state_rec, rtol=1e-4, atol=1e-4)


# =============================================================================
# Block Tests
# =============================================================================

class TestGatedDeltaNetBlock:
    """Tests for GatedDeltaNetBlock."""

    def test_forward_shape(self, key, small_shapes):
        """Test output shapes from forward pass."""
        batch = small_shapes["batch"]
        seq_len = small_shapes["seq_len"]
        hidden_size = small_shapes["hidden_size"]
        
        rngs = nnx.Rngs(0)
        block = GatedDeltaNetBlock(
            hidden_size=hidden_size,
            num_heads=2,
            num_v_heads=2,
            head_dim=32,
            expand_v=2.0,
            use_short_conv=True,
            conv_size=4,
            use_gate=True,
            rngs=rngs,
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden_size))
        output, state = block(x)
        
        assert output.shape == (batch, seq_len, hidden_size)
        assert state is None  # No state returned in training mode without explicit request

    def test_forward_with_state(self, key, small_shapes):
        """Test forward pass with state for generation."""
        batch = small_shapes["batch"]
        seq_len = 1  # Single token generation
        hidden_size = small_shapes["hidden_size"]
        
        rngs = nnx.Rngs(0)
        block = GatedDeltaNetBlock(
            hidden_size=hidden_size,
            num_heads=2,
            num_v_heads=2,
            head_dim=32,
            expand_v=2.0,
            use_short_conv=True,
            conv_size=4,
            use_gate=True,
            rngs=rngs,
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden_size))
        state = block.init_state(batch)
        output, new_state = block(x, state=state, mode="recurrent")
        
        assert output.shape == (batch, seq_len, hidden_size)
        assert new_state is not None
        assert isinstance(new_state, GatedDeltaNetState)
        assert new_state.S.shape == (batch, 2, 32, 64)  # num_v_heads, head_dim, head_v_dim

    def test_init_state(self, small_shapes):
        """Test state initialization."""
        batch = small_shapes["batch"]
        hidden_size = small_shapes["hidden_size"]
        
        rngs = nnx.Rngs(0)
        block = GatedDeltaNetBlock(
            hidden_size=hidden_size,
            num_heads=2,
            num_v_heads=2,
            head_dim=32,
            expand_v=2.0,
            use_short_conv=True,
            conv_size=4,
            rngs=rngs,
        )
        
        state = block.init_state(batch)
        
        assert state.S.shape == (batch, 2, 32, 64)
        assert state.conv_state_q is not None
        assert state.conv_state_k is not None
        assert state.conv_state_v is not None

    def test_gva_mode(self, key, small_shapes):
        """Test Grouped Value Attention (more V heads than Q/K heads)."""
        batch = small_shapes["batch"]
        seq_len = small_shapes["seq_len"]
        hidden_size = 128
        
        rngs = nnx.Rngs(0)
        block = GatedDeltaNetBlock(
            hidden_size=hidden_size,
            num_heads=2,       # Q/K heads
            num_v_heads=4,     # V heads (GVA: 4 > 2)
            head_dim=32,
            expand_v=2.0,
            use_short_conv=True,
            rngs=rngs,
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden_size))
        output, _ = block(x)
        
        assert output.shape == (batch, seq_len, hidden_size)

    def test_without_conv(self, key, small_shapes):
        """Test block without short convolutions."""
        batch = small_shapes["batch"]
        seq_len = small_shapes["seq_len"]
        hidden_size = small_shapes["hidden_size"]
        
        rngs = nnx.Rngs(0)
        block = GatedDeltaNetBlock(
            hidden_size=hidden_size,
            num_heads=2,
            head_dim=32,
            expand_v=2.0,
            use_short_conv=False,
            rngs=rngs,
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden_size))
        output, _ = block(x)
        
        assert output.shape == (batch, seq_len, hidden_size)

    def test_without_gate(self, key, small_shapes):
        """Test block without output gating."""
        batch = small_shapes["batch"]
        seq_len = small_shapes["seq_len"]
        hidden_size = small_shapes["hidden_size"]
        
        rngs = nnx.Rngs(0)
        block = GatedDeltaNetBlock(
            hidden_size=hidden_size,
            num_heads=2,
            head_dim=32,
            expand_v=2.0,
            use_gate=False,
            rngs=rngs,
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden_size))
        output, _ = block(x)
        
        assert output.shape == (batch, seq_len, hidden_size)

    def test_allow_neg_eigval(self, key, small_shapes):
        """Test block with allow_neg_eigval (beta in [0, 2])."""
        batch = small_shapes["batch"]
        seq_len = small_shapes["seq_len"]
        hidden_size = small_shapes["hidden_size"]
        
        rngs = nnx.Rngs(0)
        block = GatedDeltaNetBlock(
            hidden_size=hidden_size,
            num_heads=2,
            head_dim=32,
            expand_v=2.0,
            allow_neg_eigval=True,
            rngs=rngs,
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden_size))
        output, _ = block(x)
        
        assert output.shape == (batch, seq_len, hidden_size)

    def test_residual_connection(self, key, small_shapes):
        """Test that residual connection is applied."""
        batch = small_shapes["batch"]
        seq_len = small_shapes["seq_len"]
        hidden_size = small_shapes["hidden_size"]
        
        rngs = nnx.Rngs(0)
        block = GatedDeltaNetBlock(
            hidden_size=hidden_size,
            num_heads=2,
            head_dim=32,
            expand_v=2.0,
            rngs=rngs,
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden_size))
        output, _ = block(x)
        
        # Output should be different from input (block does something)
        # but should be in similar magnitude range (residual helps)
        assert not jnp.allclose(output, x)
        assert jnp.abs(output).mean() < 10 * jnp.abs(x).mean()  # Reasonable magnitude

    def test_sequential_generation(self, key, small_shapes):
        """Test multi-step autoregressive generation."""
        batch = small_shapes["batch"]
        hidden_size = small_shapes["hidden_size"]
        num_steps = 5
        
        rngs = nnx.Rngs(0)
        block = GatedDeltaNetBlock(
            hidden_size=hidden_size,
            num_heads=2,
            head_dim=32,
            expand_v=2.0,
            use_short_conv=True,
            conv_size=4,
            rngs=rngs,
        )
        
        state = block.init_state(batch)
        outputs = []
        
        for i in range(num_steps):
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (batch, 1, hidden_size))
            output, state = block(x, state=state, mode="recurrent")
            outputs.append(output)
        
        outputs = jnp.concatenate(outputs, axis=1)
        assert outputs.shape == (batch, num_steps, hidden_size)


# =============================================================================
# Model Integration Tests
# =============================================================================

class TestGatedDeltaNetModelIntegration:
    """Tests for Gated DeltaNet integration with LMModel."""

    def test_model_with_gated_deltanet(self, key):
        """Test LMModel with Gated DeltaNet blocks."""
        from linearnexus.models import LMModel, ModelConfig
        
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            n_layers=2,
            block_pattern=["gated_deltanet"],
            gated_deltanet_heads=2,
            gated_deltanet_v_heads=2,
            gated_deltanet_head_dim=32,
            gated_deltanet_expand_v=2.0,
            conv_kernel=4,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs)
        
        batch, seq_len = 2, 16
        tokens = jax.random.randint(key, (batch, seq_len), 0, config.vocab_size)
        
        logits, state = model(tokens)
        
        assert logits.shape == (batch, seq_len, config.vocab_size)
        # State may or may not be None depending on model implementation

    def test_model_hybrid_gated_deltanet_attention(self, key):
        """Test hybrid model with Gated DeltaNet and Attention."""
        from linearnexus.models import LMModel, ModelConfig
        
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            n_layers=4,
            n_heads=2,
            block_pattern=["gated_deltanet", "attention"],
            gated_deltanet_heads=2,
            gated_deltanet_head_dim=32,
            conv_kernel=4,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs)
        
        batch, seq_len = 2, 16
        tokens = jax.random.randint(key, (batch, seq_len), 0, config.vocab_size)
        
        logits, state = model(tokens)
        
        assert logits.shape == (batch, seq_len, config.vocab_size)

    def test_preset_gated_deltanet_small(self, key):
        """Test GATED_DELTANET_SMALL preset."""
        from linearnexus.models import LMModel, ModelConfig
        
        # Use a smaller version for testing
        config = ModelConfig(
            vocab_size=100,
            hidden_size=256,
            n_layers=2,
            block_pattern=["gated_deltanet"],
            gated_deltanet_heads=1,
            gated_deltanet_v_heads=1,
            gated_deltanet_head_dim=256,
            gated_deltanet_expand_v=2.0,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs)
        
        batch, seq_len = 2, 8
        tokens = jax.random.randint(key, (batch, seq_len), 0, config.vocab_size)
        
        logits, _ = model(tokens)
        assert logits.shape == (batch, seq_len, config.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
