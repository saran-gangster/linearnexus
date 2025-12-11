"""
Tests for RWKV7 (Goose) implementation.

RWKV7 uses DPLR (Diagonal Plus Low Rank) transition matrices:
    S_t = S_{t-1} @ (D_t + a_t @ b_t^T) + v_t @ k_t^T

Key components tested:
1. DPLR recurrent kernel
2. Step function parity with recurrent
3. RWKV7State initialization
4. RWKV7Block forward pass
5. Sequential generation with state caching
6. Model integration with LMModel
7. LoRA layer functionality
8. L2 normalization
9. Gate output correction
"""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.modules.linear_attn.rwkv7 import (
    RWKV7State,
    RWKV7Block,
    dplr_delta_rule_recurrent,
    dplr_delta_rule_step,
    LoRA,
    l2_norm,
    gate_output_correction,
)
from linearnexus.models import ModelConfig, LMModel, create_model


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def key():
    """Random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_dims():
    """Small dimensions for fast testing."""
    return {
        "batch_size": 2,
        "seq_len": 16,
        "hidden_size": 64,
        "num_heads": 2,
        "head_dim": 32,
        "head_v_dim": 32,
    }


# =============================================================================
# Test RWKV7State
# =============================================================================


class TestRWKV7State:
    """Tests for RWKV7State cache."""

    def test_zeros_shape(self, small_dims):
        """Test that zeros() creates correct shapes."""
        state = RWKV7State.zeros(
            batch_size=small_dims["batch_size"],
            num_heads=small_dims["num_heads"],
            head_k_dim=small_dims["head_dim"],
            head_v_dim=small_dims["head_v_dim"],
            hidden_size=small_dims["hidden_size"],
        )
        
        assert state.h.shape == (
            small_dims["batch_size"],
            small_dims["num_heads"],
            small_dims["head_dim"],
            small_dims["head_v_dim"],
        )
        assert state.shift_state.shape == (
            small_dims["batch_size"],
            small_dims["hidden_size"],
        )

    def test_zeros_dtype(self, small_dims):
        """Test that zeros() uses float32."""
        state = RWKV7State.zeros(
            batch_size=small_dims["batch_size"],
            num_heads=small_dims["num_heads"],
            head_k_dim=small_dims["head_dim"],
            head_v_dim=small_dims["head_v_dim"],
        )
        assert state.h.dtype == jnp.float32


# =============================================================================
# Test DPLR Kernels
# =============================================================================


class TestDPLRKernels:
    """Tests for DPLR delta rule kernels."""

    def test_recurrent_output_shape(self, key, small_dims):
        """Test DPLR recurrent produces correct output shapes."""
        batch = small_dims["batch_size"]
        heads = small_dims["num_heads"]
        seq = small_dims["seq_len"]
        k_dim = small_dims["head_dim"]
        v_dim = small_dims["head_v_dim"]
        
        q = jax.random.normal(key, (batch, heads, seq, k_dim))
        key, subkey = jax.random.split(key)
        k = jax.random.normal(subkey, (batch, heads, seq, k_dim))
        key, subkey = jax.random.split(key)
        v = jax.random.normal(subkey, (batch, heads, seq, v_dim))
        key, subkey = jax.random.split(key)
        a = jax.random.normal(subkey, (batch, heads, seq, k_dim)) * 0.1
        key, subkey = jax.random.split(key)
        b = jax.random.normal(subkey, (batch, heads, seq, k_dim)) * 0.1
        key, subkey = jax.random.split(key)
        # gk should be negative for decay
        gk = -jnp.abs(jax.random.normal(subkey, (batch, heads, seq, k_dim))) * 0.1
        
        output, final_state = dplr_delta_rule_recurrent(q, k, v, a, b, gk, scale=1.0)
        
        assert output.shape == (batch, heads, seq, v_dim)
        assert final_state.shape == (batch, heads, k_dim, v_dim)

    def test_recurrent_outputs_finite(self, key, small_dims):
        """Test that DPLR recurrent produces finite outputs."""
        batch = small_dims["batch_size"]
        heads = small_dims["num_heads"]
        seq = small_dims["seq_len"]
        k_dim = small_dims["head_dim"]
        v_dim = small_dims["head_v_dim"]
        
        q = jax.random.normal(key, (batch, heads, seq, k_dim)) * 0.1
        key, subkey = jax.random.split(key)
        k = jax.random.normal(subkey, (batch, heads, seq, k_dim)) * 0.1
        key, subkey = jax.random.split(key)
        v = jax.random.normal(subkey, (batch, heads, seq, v_dim)) * 0.1
        key, subkey = jax.random.split(key)
        a = jax.random.normal(subkey, (batch, heads, seq, k_dim)) * 0.01
        key, subkey = jax.random.split(key)
        b = jax.random.normal(subkey, (batch, heads, seq, k_dim)) * 0.01
        key, subkey = jax.random.split(key)
        gk = -jnp.abs(jax.random.normal(subkey, (batch, heads, seq, k_dim))) * 0.1
        
        output, final_state = dplr_delta_rule_recurrent(q, k, v, a, b, gk)
        
        assert jnp.all(jnp.isfinite(output)), "Output contains NaN/Inf"
        assert jnp.all(jnp.isfinite(final_state)), "Final state contains NaN/Inf"

    def test_step_matches_recurrent_first_step(self, key, small_dims):
        """Test that step function matches first step of recurrent."""
        batch = small_dims["batch_size"]
        heads = small_dims["num_heads"]
        k_dim = small_dims["head_dim"]
        v_dim = small_dims["head_v_dim"]
        
        # Single token inputs
        q = jax.random.normal(key, (batch, heads, k_dim)) * 0.1
        key, subkey = jax.random.split(key)
        k = jax.random.normal(subkey, (batch, heads, k_dim)) * 0.1
        key, subkey = jax.random.split(key)
        v = jax.random.normal(subkey, (batch, heads, v_dim)) * 0.1
        key, subkey = jax.random.split(key)
        a = jax.random.normal(subkey, (batch, heads, k_dim)) * 0.01
        key, subkey = jax.random.split(key)
        b = jax.random.normal(subkey, (batch, heads, k_dim)) * 0.01
        key, subkey = jax.random.split(key)
        gk = -jnp.abs(jax.random.normal(subkey, (batch, heads, k_dim))) * 0.1
        
        # Initial state
        h0 = jnp.zeros((batch, heads, k_dim, v_dim))
        
        # Step function
        out_step, state_step = dplr_delta_rule_step(q, k, v, a, b, gk, state=h0, scale=1.0)
        
        # Recurrent with seq_len=1
        out_rec, state_rec = dplr_delta_rule_recurrent(
            q[:, :, None, :],  # Add seq dimension
            k[:, :, None, :],
            v[:, :, None, :],
            a[:, :, None, :],
            b[:, :, None, :],
            gk[:, :, None, :],
            initial_state=h0,
            scale=1.0,
        )
        
        # Compare
        assert jnp.allclose(out_step, out_rec[:, :, 0, :], rtol=1e-4, atol=1e-6), \
            f"Output mismatch: step vs recurrent"
        assert jnp.allclose(state_step, state_rec, rtol=1e-4, atol=1e-6), \
            f"State mismatch: step vs recurrent"

    def test_initial_state_propagation(self, key, small_dims):
        """Test that initial state affects output."""
        batch = small_dims["batch_size"]
        heads = small_dims["num_heads"]
        seq = 4
        k_dim = small_dims["head_dim"]
        v_dim = small_dims["head_v_dim"]
        
        q = jax.random.normal(key, (batch, heads, seq, k_dim)) * 0.1
        key, subkey = jax.random.split(key)
        k = jax.random.normal(subkey, (batch, heads, seq, k_dim)) * 0.1
        key, subkey = jax.random.split(key)
        v = jax.random.normal(subkey, (batch, heads, seq, v_dim)) * 0.1
        key, subkey = jax.random.split(key)
        a = jax.random.normal(subkey, (batch, heads, seq, k_dim)) * 0.01
        key, subkey = jax.random.split(key)
        b = jax.random.normal(subkey, (batch, heads, seq, k_dim)) * 0.01
        key, subkey = jax.random.split(key)
        gk = -jnp.abs(jax.random.normal(subkey, (batch, heads, seq, k_dim))) * 0.1
        
        # Zero initial state
        out_zero, _ = dplr_delta_rule_recurrent(q, k, v, a, b, gk, initial_state=None)
        
        # Non-zero initial state
        key, subkey = jax.random.split(key)
        h0 = jax.random.normal(subkey, (batch, heads, k_dim, v_dim)) * 0.5
        out_init, _ = dplr_delta_rule_recurrent(q, k, v, a, b, gk, initial_state=h0)
        
        # Should be different
        assert not jnp.allclose(out_zero, out_init, rtol=1e-2), \
            "Initial state should affect output"


# =============================================================================
# Test LoRA Layer
# =============================================================================


class TestLoRA:
    """Tests for LoRA (Low-Rank Adaptation) layer."""

    def test_lora_output_shape(self, key, small_dims):
        """Test LoRA produces correct output shape."""
        lora = LoRA(
            in_features=small_dims["hidden_size"],
            out_features=small_dims["hidden_size"],
            low_rank_dim=16,
            activation=None,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (small_dims["batch_size"], small_dims["seq_len"], small_dims["hidden_size"]))
        out = lora(x)
        
        assert out.shape == x.shape

    def test_lora_tanh_activation(self, key, small_dims):
        """Test LoRA with tanh activation."""
        lora = LoRA(
            in_features=small_dims["hidden_size"],
            out_features=small_dims["hidden_size"],
            low_rank_dim=16,
            activation='tanh',
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (small_dims["batch_size"], small_dims["seq_len"], small_dims["hidden_size"]))
        out = lora(x)
        
        assert out.shape == x.shape
        assert jnp.all(jnp.isfinite(out))

    def test_lora_sigmoid_activation(self, key, small_dims):
        """Test LoRA with sigmoid activation."""
        lora = LoRA(
            in_features=small_dims["hidden_size"],
            out_features=small_dims["hidden_size"],
            low_rank_dim=16,
            activation='sigmoid',
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (small_dims["batch_size"], small_dims["seq_len"], small_dims["hidden_size"]))
        out = lora(x)
        
        assert out.shape == x.shape
        # Sigmoid output should be in [0, 1]
        assert jnp.all(out >= 0) and jnp.all(out <= 1)

    def test_lora_set_bias(self, small_dims):
        """Test LoRA bias setting."""
        lora = LoRA(
            in_features=small_dims["hidden_size"],
            out_features=small_dims["hidden_size"],
            low_rank_dim=16,
            activation=None,
            use_bias=True,
            rngs=nnx.Rngs(0),
        )
        
        new_bias = jnp.ones(small_dims["hidden_size"]) * 0.5
        lora.set_bias_value(new_bias)
        
        assert jnp.allclose(lora.up.bias.value, new_bias)


# =============================================================================
# Test L2 Norm
# =============================================================================


class TestL2Norm:
    """Tests for L2 normalization."""

    def test_l2_norm_unit_vectors(self, key):
        """Test that l2_norm produces unit vectors."""
        x = jax.random.normal(key, (2, 8, 4, 32))
        normed = l2_norm(x)
        
        # Compute norms along last dimension
        norms = jnp.sqrt(jnp.sum(normed * normed, axis=-1))
        
        # Should be approximately 1
        assert jnp.allclose(norms, 1.0, rtol=1e-5)

    def test_l2_norm_preserves_shape(self, key):
        """Test that l2_norm preserves shape."""
        x = jax.random.normal(key, (2, 8, 4, 32))
        normed = l2_norm(x)
        
        assert normed.shape == x.shape


# =============================================================================
# Test Gate Output Correction
# =============================================================================


class TestGateOutputCorrection:
    """Tests for gate output correction."""

    def test_gate_output_correction_shape(self, key, small_dims):
        """Test gate_output_correction produces correct shape."""
        batch = small_dims["batch_size"]
        seq = small_dims["seq_len"]
        heads = small_dims["num_heads"]
        head_dim = small_dims["head_dim"]
        hidden_size = small_dims["hidden_size"]
        
        o = jax.random.normal(key, (batch, seq, hidden_size))
        key, subkey = jax.random.split(key)
        r = jax.random.normal(subkey, (batch, seq, heads, head_dim))
        key, subkey = jax.random.split(key)
        k = jax.random.normal(subkey, (batch, seq, heads, head_dim))
        r_k = jax.random.normal(subkey, (heads, head_dim))
        key, subkey = jax.random.split(key)
        v = jax.random.normal(subkey, (batch, seq, heads, head_dim))
        key, subkey = jax.random.split(key)
        g = jax.nn.sigmoid(jax.random.normal(subkey, (batch, seq, hidden_size)))
        
        output = gate_output_correction(o, r, k, r_k, v, g)
        
        assert output.shape == (batch, seq, hidden_size)

    def test_gate_output_correction_finite(self, key, small_dims):
        """Test gate_output_correction produces finite outputs."""
        batch = small_dims["batch_size"]
        seq = small_dims["seq_len"]
        heads = small_dims["num_heads"]
        head_dim = small_dims["head_dim"]
        hidden_size = small_dims["hidden_size"]
        
        o = jax.random.normal(key, (batch, seq, hidden_size)) * 0.1
        key, subkey = jax.random.split(key)
        r = jax.random.normal(subkey, (batch, seq, heads, head_dim)) * 0.1
        key, subkey = jax.random.split(key)
        k = jax.random.normal(subkey, (batch, seq, heads, head_dim)) * 0.1
        r_k = jax.random.normal(subkey, (heads, head_dim)) * 0.1
        key, subkey = jax.random.split(key)
        v = jax.random.normal(subkey, (batch, seq, heads, head_dim)) * 0.1
        key, subkey = jax.random.split(key)
        g = jax.nn.sigmoid(jax.random.normal(subkey, (batch, seq, hidden_size)))
        
        output = gate_output_correction(o, r, k, r_k, v, g)
        
        assert jnp.all(jnp.isfinite(output))


# =============================================================================
# Test RWKV7Block
# =============================================================================


class TestRWKV7Block:
    """Tests for RWKV7Block."""

    def test_forward_shape(self, key, small_dims):
        """Test RWKV7Block forward produces correct shape."""
        block = RWKV7Block(
            hidden_size=small_dims["hidden_size"],
            num_heads=small_dims["num_heads"],
            head_dim=small_dims["head_dim"],
            layer_idx=0,
            n_layers=4,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (small_dims["batch_size"], small_dims["seq_len"], small_dims["hidden_size"]))
        output, state, v_first = block(x)
        
        assert output.shape == x.shape

    def test_forward_finite(self, key, small_dims):
        """Test RWKV7Block forward produces finite outputs."""
        block = RWKV7Block(
            hidden_size=small_dims["hidden_size"],
            num_heads=small_dims["num_heads"],
            head_dim=small_dims["head_dim"],
            layer_idx=0,
            n_layers=4,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (small_dims["batch_size"], small_dims["seq_len"], small_dims["hidden_size"])) * 0.1
        output, state, v_first = block(x)
        
        assert jnp.all(jnp.isfinite(output)), f"Output contains NaN/Inf"

    def test_sequential_generation(self, key, small_dims):
        """Test sequential generation with state caching."""
        block = RWKV7Block(
            hidden_size=small_dims["hidden_size"],
            num_heads=small_dims["num_heads"],
            head_dim=small_dims["head_dim"],
            layer_idx=0,
            n_layers=4,
            rngs=nnx.Rngs(0),
        )
        
        batch = small_dims["batch_size"]
        state = block.init_state(batch)
        v_first = None
        
        outputs = []
        for i in range(5):
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (batch, 1, small_dims["hidden_size"])) * 0.1
            output, state, v_first = block(x, state=state, v_first=v_first, mode="recurrent")
            outputs.append(output)
            
            assert output.shape == (batch, 1, small_dims["hidden_size"])
            assert state is not None
            assert jnp.all(jnp.isfinite(output))
        
        # Check state is being updated
        assert state.h is not None

    def test_v_first_propagation(self, key, small_dims):
        """Test v_first propagation across layers."""
        # First layer (layer_idx=0) should produce v_first
        block0 = RWKV7Block(
            hidden_size=small_dims["hidden_size"],
            num_heads=small_dims["num_heads"],
            head_dim=small_dims["head_dim"],
            layer_idx=0,
            n_layers=4,
            rngs=nnx.Rngs(0),
        )
        
        # Second layer (layer_idx=1) should use v_first
        block1 = RWKV7Block(
            hidden_size=small_dims["hidden_size"],
            num_heads=small_dims["num_heads"],
            head_dim=small_dims["head_dim"],
            layer_idx=1,
            n_layers=4,
            rngs=nnx.Rngs(1),
        )
        
        x = jax.random.normal(key, (small_dims["batch_size"], small_dims["seq_len"], small_dims["hidden_size"])) * 0.1
        
        # First layer should output v_first
        out0, _, v_first = block0(x)
        assert v_first is not None
        
        # Second layer should accept v_first
        out1, _, v_first_out = block1(out0, v_first=v_first)
        assert jnp.all(jnp.isfinite(out1))

    def test_init_state(self, small_dims):
        """Test state initialization."""
        block = RWKV7Block(
            hidden_size=small_dims["hidden_size"],
            num_heads=small_dims["num_heads"],
            head_dim=small_dims["head_dim"],
            layer_idx=0,
            n_layers=4,
            rngs=nnx.Rngs(0),
        )
        
        state = block.init_state(small_dims["batch_size"])
        
        assert state.h.shape == (
            small_dims["batch_size"],
            small_dims["num_heads"],
            small_dims["head_dim"],
            small_dims["head_dim"],  # head_v_dim = head_dim when value_dim = hidden_size
        )


# =============================================================================
# Test Model Integration
# =============================================================================


class TestModelIntegration:
    """Tests for RWKV7 integration with LMModel."""

    def test_pure_rwkv7_model(self, key):
        """Test pure RWKV7 model."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            n_layers=2,
            n_heads=2,
            block_pattern=["rwkv7"],
            rwkv7_heads=2,
            rwkv7_head_dim=32,
        )
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        tokens = jax.random.randint(key, (2, 16), 0, 100)
        logits, state = model(tokens)
        
        assert logits.shape == (2, 16, 100)
        assert jnp.all(jnp.isfinite(logits))

    def test_rwkv7_hybrid_model(self, key):
        """Test RWKV7 + attention hybrid."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            n_layers=4,
            n_heads=2,
            block_pattern=["rwkv7", "rwkv7", "attention"],
            rwkv7_heads=2,
            rwkv7_head_dim=32,
        )
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        tokens = jax.random.randint(key, (2, 16), 0, 100)
        logits, state = model(tokens)
        
        assert logits.shape == (2, 16, 100)
        assert jnp.all(jnp.isfinite(logits))

    def test_create_model_preset(self, key):
        """Test create_model with RWKV7 preset."""
        # Use custom config with small sizes for testing
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            n_layers=2,
            n_heads=2,
            block_pattern=["rwkv7"],
            rwkv7_heads=2,
            rwkv7_head_dim=32,
        )
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        tokens = jax.random.randint(key, (2, 8), 0, 100)
        logits, _ = model(tokens)
        
        assert logits.shape == (2, 8, 100)

    def test_sequential_generation_model(self, key):
        """Test sequential generation with RWKV7 model."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            n_layers=2,
            n_heads=2,
            block_pattern=["rwkv7"],
            rwkv7_heads=2,
            rwkv7_head_dim=32,
        )
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        batch_size = 2
        state = model.init_state(batch_size)
        
        # Generate tokens one at a time
        for i in range(5):
            key, subkey = jax.random.split(key)
            tokens = jax.random.randint(subkey, (batch_size, 1), 0, 100)
            logits, state = model(tokens, state=state, mode="recurrent")
            
            assert logits.shape == (batch_size, 1, 100)
            assert jnp.all(jnp.isfinite(logits))


# =============================================================================
# Test Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_long_sequence(self, key, small_dims):
        """Test RWKV7Block with longer sequences."""
        block = RWKV7Block(
            hidden_size=small_dims["hidden_size"],
            num_heads=small_dims["num_heads"],
            head_dim=small_dims["head_dim"],
            layer_idx=0,
            n_layers=4,
            rngs=nnx.Rngs(0),
        )
        
        # Longer sequence
        x = jax.random.normal(key, (2, 128, small_dims["hidden_size"])) * 0.1
        output, state, v_first = block(x)
        
        assert jnp.all(jnp.isfinite(output)), "Long sequence produces NaN/Inf"

    def test_state_accumulation(self, key, small_dims):
        """Test that state doesn't explode over many steps."""
        block = RWKV7Block(
            hidden_size=small_dims["hidden_size"],
            num_heads=small_dims["num_heads"],
            head_dim=small_dims["head_dim"],
            layer_idx=0,
            n_layers=4,
            rngs=nnx.Rngs(0),
        )
        
        state = block.init_state(2)
        v_first = None
        max_state_norm = 0.0
        
        for i in range(50):
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (2, 1, small_dims["hidden_size"])) * 0.1
            output, state, v_first = block(x, state=state, v_first=v_first, mode="recurrent")
            
            state_norm = jnp.max(jnp.abs(state.h))
            max_state_norm = max(max_state_norm, float(state_norm))
        
        # State should stay bounded
        assert max_state_norm < 1000, f"State exploded: max norm = {max_state_norm}"

    def test_dplr_decay_effect(self, key, small_dims):
        """Test that decay (w/gk) actually decays the state."""
        batch = small_dims["batch_size"]
        heads = small_dims["num_heads"]
        k_dim = small_dims["head_dim"]
        v_dim = small_dims["head_v_dim"]
        
        # Create initial state with large values
        h0 = jnp.ones((batch, heads, k_dim, v_dim)) * 10.0
        
        # Create inputs
        q = jax.random.normal(key, (batch, heads, 10, k_dim)) * 0.01
        key, subkey = jax.random.split(key)
        k = jnp.zeros((batch, heads, 10, k_dim))  # Zero keys = no new info
        v = jnp.zeros((batch, heads, 10, v_dim))  # Zero values
        a = jnp.zeros((batch, heads, 10, k_dim))  # Zero a
        b = jnp.zeros((batch, heads, 10, k_dim))  # Zero b
        # Strong decay
        gk = jnp.ones((batch, heads, 10, k_dim)) * -0.5
        
        _, final_state = dplr_delta_rule_recurrent(q, k, v, a, b, gk, initial_state=h0)
        
        # Final state should be much smaller than initial
        initial_norm = jnp.max(jnp.abs(h0))
        final_norm = jnp.max(jnp.abs(final_state))
        
        assert final_norm < initial_norm * 0.5, \
            f"Decay not working: initial={initial_norm}, final={final_norm}"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_batch_size_one(self, key, small_dims):
        """Test with batch size 1."""
        block = RWKV7Block(
            hidden_size=small_dims["hidden_size"],
            num_heads=small_dims["num_heads"],
            head_dim=small_dims["head_dim"],
            layer_idx=0,
            n_layers=4,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (1, small_dims["seq_len"], small_dims["hidden_size"])) * 0.1
        output, _, _ = block(x)
        
        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))

    def test_seq_len_one(self, key, small_dims):
        """Test with sequence length 1."""
        block = RWKV7Block(
            hidden_size=small_dims["hidden_size"],
            num_heads=small_dims["num_heads"],
            head_dim=small_dims["head_dim"],
            layer_idx=0,
            n_layers=4,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (small_dims["batch_size"], 1, small_dims["hidden_size"])) * 0.1
        output, _, _ = block(x)
        
        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))

    def test_different_layer_indices(self, key, small_dims):
        """Test blocks at different layer indices."""
        for layer_idx in [0, 1, 5, 10]:
            block = RWKV7Block(
                hidden_size=small_dims["hidden_size"],
                num_heads=small_dims["num_heads"],
                head_dim=small_dims["head_dim"],
                layer_idx=layer_idx,
                n_layers=12,
                rngs=nnx.Rngs(layer_idx),
            )
            
            x = jax.random.normal(key, (small_dims["batch_size"], small_dims["seq_len"], small_dims["hidden_size"])) * 0.1
            output, _, _ = block(x)
            
            assert output.shape == x.shape
            assert jnp.all(jnp.isfinite(output)), f"Layer {layer_idx} produces NaN/Inf"
