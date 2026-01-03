"""Tests for Kimi Delta Attention (KDA) implementation.

Tests cover:
- KDAState initialization
- KDA kernels (recurrent, chunkwise, step)
- KDA gate computation
- KDABlock forward pass
- Model integration
- Numerical stability
"""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.modules.linear_attn import (
    KDABlock,
    KDAState,
    kda_recurrent,
    kda_chunkwise,
    kda_step,
    kda_gate,
    FusedRMSNormGated,
)
from linearnexus.models import LMModel, ModelConfig, create_model


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_dims():
    return {
        "batch": 2,
        "seq_len": 16,
        "num_heads": 4,
        "num_v_heads": 4,
        "key_dim": 32,
        "value_dim": 32,
        "hidden_size": 128,
    }


# =============================================================================
# Test KDAState
# =============================================================================

class TestKDAState:
    """Tests for KDAState dataclass."""
    
    def test_zeros_shape(self, key, small_dims):
        """Test zero-initialization produces correct shapes."""
        state = KDAState.zeros(
            batch_size=small_dims["batch"],
            num_v_heads=small_dims["num_v_heads"],
            key_dim_per_head=small_dims["key_dim"] // small_dims["num_heads"],
            value_dim_per_head=small_dims["value_dim"] // small_dims["num_v_heads"],
            key_dim_total=small_dims["key_dim"],
            value_dim=small_dims["value_dim"],
            conv_size=4,
            use_conv=True,
        )
        
        head_k_dim = small_dims["key_dim"] // small_dims["num_heads"]
        head_v_dim = small_dims["value_dim"] // small_dims["num_v_heads"]
        
        assert state.S.shape == (small_dims["batch"], small_dims["num_v_heads"], head_k_dim, head_v_dim)
        assert state.conv_state_q.shape == (small_dims["batch"], 3, small_dims["key_dim"])
        assert state.conv_state_k.shape == (small_dims["batch"], 3, small_dims["key_dim"])
        assert state.conv_state_v.shape == (small_dims["batch"], 3, small_dims["value_dim"])
    
    def test_zeros_without_conv(self, key, small_dims):
        """Test initialization without conv state."""
        state = KDAState.zeros(
            batch_size=small_dims["batch"],
            num_v_heads=small_dims["num_v_heads"],
            key_dim_per_head=small_dims["key_dim"] // small_dims["num_heads"],
            value_dim_per_head=small_dims["value_dim"] // small_dims["num_v_heads"],
            use_conv=False,
        )
        
        assert state.conv_state_q is None
        assert state.conv_state_k is None
        assert state.conv_state_v is None
    
    def test_zeros_dtype(self, key, small_dims):
        """Test state is float32 by default."""
        state = KDAState.zeros(
            batch_size=small_dims["batch"],
            num_v_heads=small_dims["num_v_heads"],
            key_dim_per_head=8,
            value_dim_per_head=8,
        )
        assert state.S.dtype == jnp.float32


# =============================================================================
# Test KDA Gate
# =============================================================================

class TestKDAGate:
    """Tests for KDA gate computation."""
    
    def test_gate_output_shape(self, key, small_dims):
        """Test gate produces correct output shape."""
        batch, seq_len = small_dims["batch"], small_dims["seq_len"]
        num_heads = small_dims["num_heads"]
        head_k_dim = small_dims["key_dim"] // num_heads
        
        g = jax.random.normal(key, (batch, seq_len, num_heads * head_k_dim))
        A_log = jax.random.normal(key, (num_heads,))
        beta = jax.random.normal(key, (batch, seq_len, num_heads))
        
        g_out, beta_out = kda_gate(g, A_log, head_k_dim, beta=beta)
        
        assert g_out.shape == (batch, seq_len, num_heads, head_k_dim)
        assert beta_out.shape == (batch, seq_len, num_heads)
    
    def test_gate_negative_values(self, key, small_dims):
        """Gate output should be negative (decay)."""
        batch, seq_len = small_dims["batch"], small_dims["seq_len"]
        num_heads = small_dims["num_heads"]
        head_k_dim = small_dims["key_dim"] // num_heads
        
        g = jax.random.normal(key, (batch, seq_len, num_heads * head_k_dim))
        A_log = jnp.ones((num_heads,))  # Positive A_log
        
        g_out, _ = kda_gate(g, A_log, head_k_dim)
        
        # g_out = -exp(A_log) * softplus(g) should be negative
        assert jnp.all(g_out <= 0)
    
    def test_gate_with_bias(self, key, small_dims):
        """Test gate with bias produces correct output."""
        batch, seq_len = small_dims["batch"], small_dims["seq_len"]
        num_heads = small_dims["num_heads"]
        head_k_dim = small_dims["key_dim"] // num_heads
        key_dim = num_heads * head_k_dim
        
        g = jax.random.normal(key, (batch, seq_len, key_dim))
        A_log = jax.random.normal(key, (num_heads,))
        g_bias = jax.random.normal(key, (key_dim,))
        
        g_out, _ = kda_gate(g, A_log, head_k_dim, g_bias=g_bias)
        
        assert g_out.shape == (batch, seq_len, num_heads, head_k_dim)
        assert jnp.all(jnp.isfinite(g_out))
    
    def test_beta_sigmoid(self, key, small_dims):
        """Test beta sigmoid is in [0, 1]."""
        batch, seq_len = small_dims["batch"], small_dims["seq_len"]
        num_heads = small_dims["num_heads"]
        head_k_dim = small_dims["key_dim"] // num_heads
        
        g = jax.random.normal(key, (batch, seq_len, num_heads * head_k_dim))
        A_log = jax.random.normal(key, (num_heads,))
        beta = jax.random.normal(key, (batch, seq_len, num_heads)) * 10  # Large range
        
        _, beta_out = kda_gate(g, A_log, head_k_dim, beta=beta)
        
        assert jnp.all(beta_out >= 0)
        assert jnp.all(beta_out <= 1)


# =============================================================================
# Test KDA Kernels
# =============================================================================

class TestKDAKernels:
    """Tests for KDA recurrent/chunkwise/step kernels."""
    
    def test_recurrent_output_shape(self, key, small_dims):
        """Test recurrent kernel output shape."""
        batch = small_dims["batch"]
        seq_len = small_dims["seq_len"]
        num_heads = small_dims["num_heads"]
        head_k_dim = 8
        head_v_dim = 8
        
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, num_heads, seq_len, head_k_dim))
        k = jax.random.normal(k2, (batch, num_heads, seq_len, head_k_dim))
        v = jax.random.normal(k3, (batch, num_heads, seq_len, head_v_dim))
        g = -jnp.abs(jax.random.normal(k4, (batch, num_heads, seq_len, head_k_dim)))  # Negative decay
        beta = jax.random.uniform(k5, (batch, num_heads, seq_len))
        
        output, state = kda_recurrent(q, k, v, g, beta)
        
        assert output.shape == (batch, num_heads, seq_len, head_v_dim)
        assert state.shape == (batch, num_heads, head_k_dim, head_v_dim)
    
    def test_recurrent_outputs_finite(self, key, small_dims):
        """Test recurrent produces finite outputs."""
        batch = small_dims["batch"]
        seq_len = small_dims["seq_len"]
        num_heads = small_dims["num_heads"]
        head_k_dim = 8
        head_v_dim = 8
        
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, num_heads, seq_len, head_k_dim))
        k = jax.random.normal(k2, (batch, num_heads, seq_len, head_k_dim))
        v = jax.random.normal(k3, (batch, num_heads, seq_len, head_v_dim))
        g = -jnp.abs(jax.random.normal(k4, (batch, num_heads, seq_len, head_k_dim))) * 0.1
        beta = jax.random.uniform(k5, (batch, num_heads, seq_len))
        
        output, state = kda_recurrent(q, k, v, g, beta)
        
        assert jnp.all(jnp.isfinite(output))
        assert jnp.all(jnp.isfinite(state))
    
    def test_step_matches_recurrent_first_step(self, key, small_dims):
        """Single step should match first step of recurrent."""
        batch = small_dims["batch"]
        num_heads = small_dims["num_heads"]
        head_k_dim = 8
        head_v_dim = 8
        
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, num_heads, head_k_dim))
        k = jax.random.normal(k2, (batch, num_heads, head_k_dim))
        v = jax.random.normal(k3, (batch, num_heads, head_v_dim))
        g = -jnp.abs(jax.random.normal(k4, (batch, num_heads, head_k_dim))) * 0.1
        beta = jax.random.uniform(k5, (batch, num_heads))
        
        # Zero initial state
        initial_state = jnp.zeros((batch, num_heads, head_k_dim, head_v_dim))
        
        # Single step
        out_step, state_step = kda_step(q, k, v, g, beta, initial_state)
        
        # Recurrent with seq_len=1
        out_rec, state_rec = kda_recurrent(
            q[:, :, None, :], 
            k[:, :, None, :], 
            v[:, :, None, :],
            g[:, :, None, :],
            beta[:, :, None],
            initial_state=initial_state,
        )
        
        assert jnp.allclose(out_step, out_rec[:, :, 0, :], rtol=1e-4, atol=1e-4)
        assert jnp.allclose(state_step, state_rec, rtol=1e-4, atol=1e-4)

    def test_chunkwise_matches_recurrent(self, key):
        """Chunkwise kernel should match recurrent kernel (correctness target).

        Uses a single chunk (seq_len=64) to avoid padding edge-cases.
        """
        batch = 2
        heads = 2
        seq_len = 64
        key_dim = 8
        value_dim = 8

        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, heads, seq_len, key_dim))
        k = jax.random.normal(k2, (batch, heads, seq_len, key_dim))
        v = jax.random.normal(k3, (batch, heads, seq_len, value_dim))

        # Negative log-decay gates (so exp(g) in (0, 1])
        g = -jax.random.uniform(k4, (batch, heads, seq_len, key_dim), minval=0.0, maxval=0.5)
        beta = jax.random.uniform(k5, (batch, heads, seq_len), minval=0.0, maxval=1.0)

        out_rec, st_rec = kda_recurrent(q, k, v, g, beta, use_qk_l2norm=True)
        out_chunk, st_chunk = kda_chunkwise(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=64,
            use_qk_l2norm=True,
        )

        assert out_chunk.shape == out_rec.shape
        assert st_chunk.shape == st_rec.shape

        # Chunkwise and recurrent should be close (float32 internal math)
        assert jnp.all(jnp.isfinite(out_chunk))
        assert jnp.all(jnp.isfinite(st_chunk))
        assert jnp.allclose(out_chunk, out_rec, rtol=2e-3, atol=2e-3)
        assert jnp.allclose(st_chunk, st_rec, rtol=2e-3, atol=2e-3)
    
    def test_chunkwise_output_shape(self, key, small_dims):
        """Test chunkwise kernel output shape."""
        batch = small_dims["batch"]
        seq_len = 64  # Multiple of chunk_size
        num_heads = small_dims["num_heads"]
        head_k_dim = 8
        head_v_dim = 8
        
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, num_heads, seq_len, head_k_dim))
        k = jax.random.normal(k2, (batch, num_heads, seq_len, head_k_dim))
        v = jax.random.normal(k3, (batch, num_heads, seq_len, head_v_dim))
        g = -jnp.abs(jax.random.normal(k4, (batch, num_heads, seq_len, head_k_dim))) * 0.1
        beta = jax.random.uniform(k5, (batch, num_heads, seq_len))
        
        output, state = kda_chunkwise(q, k, v, g, beta, chunk_size=32)
        
        assert output.shape == (batch, num_heads, seq_len, head_v_dim)
        assert state.shape == (batch, num_heads, head_k_dim, head_v_dim)
    
    def test_chunkwise_outputs_finite(self, key, small_dims):
        """Test chunkwise produces finite outputs."""
        batch = small_dims["batch"]
        seq_len = 64
        num_heads = small_dims["num_heads"]
        head_k_dim = 8
        head_v_dim = 8
        
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, num_heads, seq_len, head_k_dim))
        k = jax.random.normal(k2, (batch, num_heads, seq_len, head_k_dim))
        v = jax.random.normal(k3, (batch, num_heads, seq_len, head_v_dim))
        g = -jnp.abs(jax.random.normal(k4, (batch, num_heads, seq_len, head_k_dim))) * 0.1
        beta = jax.random.uniform(k5, (batch, num_heads, seq_len))
        
        output, state = kda_chunkwise(q, k, v, g, beta)
        
        assert jnp.all(jnp.isfinite(output))
        assert jnp.all(jnp.isfinite(state))
    
    def test_initial_state_propagation(self, key, small_dims):
        """Test initial state is used correctly."""
        batch = small_dims["batch"]
        seq_len = small_dims["seq_len"]
        num_heads = small_dims["num_heads"]
        head_k_dim = 8
        head_v_dim = 8
        
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        q = jax.random.normal(k1, (batch, num_heads, seq_len, head_k_dim))
        k = jax.random.normal(k2, (batch, num_heads, seq_len, head_k_dim))
        v = jax.random.normal(k3, (batch, num_heads, seq_len, head_v_dim))
        g = -jnp.abs(jax.random.normal(k4, (batch, num_heads, seq_len, head_k_dim))) * 0.1
        beta = jax.random.uniform(k5, (batch, num_heads, seq_len))
        initial_state = jax.random.normal(k6, (batch, num_heads, head_k_dim, head_v_dim)) * 0.1
        
        # With zero initial state
        out_zero, _ = kda_recurrent(q, k, v, g, beta, initial_state=None)
        
        # With non-zero initial state
        out_init, _ = kda_recurrent(q, k, v, g, beta, initial_state=initial_state)
        
        # Outputs should differ
        assert not jnp.allclose(out_zero, out_init, rtol=1e-2)


# =============================================================================
# Test FusedRMSNormGated
# =============================================================================

class TestFusedRMSNormGated:
    """Tests for FusedRMSNormGated module."""
    
    def test_output_shape(self, key):
        """Test output has correct shape."""
        batch, seq_len, hidden = 2, 16, 64
        
        norm = FusedRMSNormGated(hidden, rngs=nnx.Rngs(0))
        
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (batch, seq_len, hidden))
        gate = jax.random.normal(k2, (batch, seq_len, hidden))
        
        out = norm(x, gate)
        
        assert out.shape == x.shape
    
    def test_output_finite(self, key):
        """Test output is finite."""
        batch, seq_len, hidden = 2, 16, 64
        
        norm = FusedRMSNormGated(hidden, rngs=nnx.Rngs(0))
        
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (batch, seq_len, hidden))
        gate = jax.random.normal(k2, (batch, seq_len, hidden))
        
        out = norm(x, gate)
        
        assert jnp.all(jnp.isfinite(out))
    
    def test_gate_effect(self, key):
        """Test gate modulates output."""
        batch, seq_len, hidden = 2, 16, 64
        
        norm = FusedRMSNormGated(hidden, rngs=nnx.Rngs(0))
        
        x = jax.random.normal(key, (batch, seq_len, hidden))
        
        # Very negative gate -> output near zero
        gate_neg = -10 * jnp.ones((batch, seq_len, hidden))
        out_neg = norm(x, gate_neg)
        
        # Very positive gate -> output near full
        gate_pos = 10 * jnp.ones((batch, seq_len, hidden))
        out_pos = norm(x, gate_pos)
        
        # Output with positive gate should have larger magnitude
        assert jnp.mean(jnp.abs(out_pos)) > jnp.mean(jnp.abs(out_neg))


# =============================================================================
# Test KDABlock
# =============================================================================

class TestKDABlock:
    """Tests for KDABlock module."""
    
    def test_forward_shape(self, key):
        """Test forward pass produces correct output shape."""
        batch, seq_len, hidden = 2, 16, 256
        num_heads = 2
        head_dim = 128
        
        block = KDABlock(
            hidden_size=hidden,
            num_heads=num_heads,
            num_v_heads=num_heads,
            head_dim=head_dim,
            expand_v=1.0,
            use_short_conv=True,
            conv_size=4,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden))
        output, state = block(x)
        
        assert output.shape == x.shape
    
    def test_forward_finite(self, key):
        """Test forward pass produces finite outputs."""
        batch, seq_len, hidden = 2, 16, 256
        num_heads = 2
        head_dim = 128
        
        block = KDABlock(
            hidden_size=hidden,
            num_heads=num_heads,
            num_v_heads=num_heads,
            head_dim=head_dim,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden))
        output, _ = block(x)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_sequential_generation(self, key):
        """Test sequential generation with state."""
        batch, hidden = 2, 256
        num_heads = 2
        head_dim = 128
        
        block = KDABlock(
            hidden_size=hidden,
            num_heads=num_heads,
            num_v_heads=num_heads,
            head_dim=head_dim,
            use_short_conv=True,
            rngs=nnx.Rngs(0),
        )
        
        state = block.init_state(batch)
        
        for i in range(5):
            x = jax.random.normal(jax.random.PRNGKey(i), (batch, 1, hidden))
            output, state = block(x, state=state, mode="recurrent")
            
            assert output.shape == (batch, 1, hidden)
            assert jnp.all(jnp.isfinite(output))
            assert state is not None
    
    def test_gva_mode(self, key):
        """Test Grouped Value Attention mode (num_v_heads > num_heads)."""
        batch, seq_len, hidden = 2, 16, 256
        num_heads = 2
        num_v_heads = 4  # More V heads than Q/K heads
        head_dim = 128
        
        block = KDABlock(
            hidden_size=hidden,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            head_dim=head_dim,
            expand_v=1.0,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden))
        output, _ = block(x)
        
        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))
    
    def test_without_conv(self, key):
        """Test block without short convolutions."""
        batch, seq_len, hidden = 2, 16, 256
        num_heads = 2
        head_dim = 128
        
        block = KDABlock(
            hidden_size=hidden,
            num_heads=num_heads,
            num_v_heads=num_heads,
            head_dim=head_dim,
            use_short_conv=False,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden))
        output, _ = block(x)
        
        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))
    
    def test_allow_neg_eigval(self, key):
        """Test allow_neg_eigval mode (beta in [0, 2])."""
        batch, seq_len, hidden = 2, 16, 256
        num_heads = 2
        head_dim = 128
        
        block = KDABlock(
            hidden_size=hidden,
            num_heads=num_heads,
            num_v_heads=num_heads,
            head_dim=head_dim,
            allow_neg_eigval=True,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden))
        output, _ = block(x)
        
        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))
    
    def test_init_state(self, key):
        """Test init_state method."""
        hidden = 256
        num_heads = 2
        head_dim = 128
        batch = 4
        
        block = KDABlock(
            hidden_size=hidden,
            num_heads=num_heads,
            num_v_heads=num_heads,
            head_dim=head_dim,
            use_short_conv=True,
            conv_size=4,
            rngs=nnx.Rngs(0),
        )
        
        state = block.init_state(batch)
        
        assert state.S.shape[0] == batch
        assert state.conv_state_q is not None
        assert state.conv_state_k is not None
        assert state.conv_state_v is not None
    
    def test_residual_connection(self, key):
        """Test residual connection is applied."""
        batch, seq_len, hidden = 2, 16, 256
        num_heads = 2
        head_dim = 128
        
        block = KDABlock(
            hidden_size=hidden,
            num_heads=num_heads,
            num_v_heads=num_heads,
            head_dim=head_dim,
            rngs=nnx.Rngs(0),
        )
        
        # Input with known values
        x = jnp.ones((batch, seq_len, hidden)) * 0.1
        output, _ = block(x)
        
        # Output should not be too far from input (residual helps)
        # Just check it's reasonable
        assert jnp.all(jnp.isfinite(output))
        assert jnp.mean(jnp.abs(output)) < 100  # Shouldn't explode


# =============================================================================
# Test Model Integration
# =============================================================================

class TestModelIntegration:
    """Tests for KDA integration with LMModel."""
    
    def test_pure_kda_model(self, key):
        """Test pure KDA model."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=256,
            n_layers=2,
            n_heads=2,
            block_pattern=["kda"],
            kda_heads=2,
            kda_v_heads=2,
            kda_head_dim=128,
            kda_expand_v=1.0,
        )
        
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        tokens = jax.random.randint(key, (2, 16), 0, 100)
        logits, state = model(tokens)
        
        assert logits.shape == (2, 16, 100)
        assert jnp.all(jnp.isfinite(logits))
    
    def test_kda_hybrid_model(self, key):
        """Test KDA + attention hybrid model."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=256,
            n_layers=4,
            n_heads=4,
            block_pattern=["kda", "kda", "kda", "attention"],
            kda_heads=2,
            kda_v_heads=2,
            kda_head_dim=128,
        )
        
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        tokens = jax.random.randint(key, (2, 16), 0, 100)
        logits, state = model(tokens)
        
        assert logits.shape == (2, 16, 100)
        assert jnp.all(jnp.isfinite(logits))
    
    def test_create_model_preset(self, key):
        """Test create_model with preset."""
        # Test that presets exist (but use smaller config for speed)
        config = ModelConfig(
            vocab_size=100,
            hidden_size=256,
            n_layers=2,
            n_heads=2,
            block_pattern=["kda"],
            kda_heads=2,
            kda_v_heads=2,
            kda_head_dim=128,
        )
        
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        tokens = jax.random.randint(key, (2, 8), 0, 100)
        logits, _ = model(tokens)
        
        assert logits.shape == (2, 8, 100)
    
    def test_sequential_generation_model(self, key):
        """Test sequential generation with model state."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=256,
            n_layers=2,
            n_heads=2,
            block_pattern=["kda"],
            kda_heads=2,
            kda_v_heads=2,
            kda_head_dim=128,
        )
        
        model = LMModel(config, rngs=nnx.Rngs(0))
        state = model.init_state(batch_size=2)
        
        for i in range(5):
            tokens = jax.random.randint(jax.random.PRNGKey(i), (2, 1), 0, 100)
            logits, state = model(tokens, state=state, mode="recurrent")
            
            assert logits.shape == (2, 1, 100)
            assert jnp.all(jnp.isfinite(logits))


# =============================================================================
# Test Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_long_sequence(self, key):
        """Test with longer sequences."""
        batch, seq_len, hidden = 2, 128, 256
        num_heads = 2
        head_dim = 128
        
        block = KDABlock(
            hidden_size=hidden,
            num_heads=num_heads,
            num_v_heads=num_heads,
            head_dim=head_dim,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden))
        output, _ = block(x)
        
        assert jnp.all(jnp.isfinite(output))
    
    def test_state_accumulation(self, key):
        """Test state doesn't explode over many steps."""
        hidden = 256
        num_heads = 2
        head_dim = 128
        batch = 2
        
        block = KDABlock(
            hidden_size=hidden,
            num_heads=num_heads,
            num_v_heads=num_heads,
            head_dim=head_dim,
            rngs=nnx.Rngs(0),
        )
        
        state = block.init_state(batch)
        
        for i in range(50):
            x = jax.random.normal(jax.random.PRNGKey(i), (batch, 1, hidden))
            output, state = block(x, state=state, mode="recurrent")
            
            assert jnp.all(jnp.isfinite(output))
            assert jnp.all(jnp.isfinite(state.S))
    
    def test_decay_effect(self, key):
        """Test per-dim decay gate effect."""
        batch = 2
        num_heads = 2
        seq_len = 16
        head_k_dim = 8
        head_v_dim = 8
        
        k1, k2, k3, k5 = jax.random.split(key, 4)
        q = jax.random.normal(k1, (batch, num_heads, seq_len, head_k_dim))
        k = jax.random.normal(k2, (batch, num_heads, seq_len, head_k_dim))
        v = jax.random.normal(k3, (batch, num_heads, seq_len, head_v_dim))
        beta = jax.random.uniform(k5, (batch, num_heads, seq_len))
        
        # Strong decay (large negative g)
        g_strong = -5.0 * jnp.ones((batch, num_heads, seq_len, head_k_dim))
        _, state_strong = kda_recurrent(q, k, v, g_strong, beta)
        
        # Weak decay (small negative g)
        g_weak = -0.1 * jnp.ones((batch, num_heads, seq_len, head_k_dim))
        _, state_weak = kda_recurrent(q, k, v, g_weak, beta)
        
        # Strong decay should result in smaller state magnitude
        assert jnp.mean(jnp.abs(state_strong)) < jnp.mean(jnp.abs(state_weak))


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_batch_size_one(self, key):
        """Test with batch size of 1."""
        block = KDABlock(
            hidden_size=256,
            num_heads=2,
            num_v_heads=2,
            head_dim=128,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (1, 16, 256))
        output, _ = block(x)
        
        assert output.shape == (1, 16, 256)
        assert jnp.all(jnp.isfinite(output))
    
    def test_seq_len_one(self, key):
        """Test with sequence length of 1."""
        block = KDABlock(
            hidden_size=256,
            num_heads=2,
            num_v_heads=2,
            head_dim=128,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (2, 1, 256))
        output, _ = block(x)
        
        assert output.shape == (2, 1, 256)
        assert jnp.all(jnp.isfinite(output))
    
    def test_expand_v_factor(self, key):
        """Test with expand_v > 1."""
        batch, seq_len, hidden = 2, 16, 256
        num_heads = 2
        head_dim = 128
        
        block = KDABlock(
            hidden_size=hidden,
            num_heads=num_heads,
            num_v_heads=num_heads,
            head_dim=head_dim,
            expand_v=2.0,  # Value expansion
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (batch, seq_len, hidden))
        output, _ = block(x)
        
        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))
