"""Tests for RWKV6 implementation.

Validates the RWKV6 kernel functions and RWKV6Block.
Based on the FLA implementation from "Eagle and Finch: RWKV with Matrix-Valued States".
"""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.modules.linear_attn.rwkv6 import (
    RWKV6Block,
    RWKV6State,
    rwkv6_recurrent,
    rwkv6_step,
    token_shift,
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
        "head_dim": 32,
    }


@pytest.fixture
def medium_shapes():
    """Medium shapes closer to real usage."""
    return {
        "batch": 2,
        "seq_len": 128,
        "hidden_size": 256,
        "num_heads": 4,
        "head_dim": 64,
    }


# =============================================================================
# State Tests
# =============================================================================

class TestRWKV6State:
    """Tests for RWKV6State cache."""

    def test_zeros_basic(self, small_shapes):
        """Test basic state initialization."""
        state = RWKV6State.zeros(
            batch_size=small_shapes["batch"],
            num_heads=small_shapes["num_heads"],
            head_k_dim=small_shapes["head_dim"],
            head_v_dim=small_shapes["head_dim"],
        )
        
        assert state.h.shape == (
            small_shapes["batch"],
            small_shapes["num_heads"],
            small_shapes["head_dim"],
            small_shapes["head_dim"],
        )
        assert jnp.allclose(state.h, 0.0)
        assert state.shift_state is None

    def test_zeros_with_shift_state(self, small_shapes):
        """Test state initialization with shift state."""
        state = RWKV6State.zeros(
            batch_size=small_shapes["batch"],
            num_heads=small_shapes["num_heads"],
            head_k_dim=small_shapes["head_dim"],
            head_v_dim=small_shapes["head_dim"],
            hidden_size=small_shapes["hidden_size"],
        )
        
        assert state.shift_state.shape == (
            small_shapes["batch"],
            small_shapes["hidden_size"],
        )
        assert jnp.allclose(state.shift_state, 0.0)


# =============================================================================
# Token Shift Tests
# =============================================================================

class TestTokenShift:
    """Tests for token shift helper."""

    def test_token_shift_basic(self, key, small_shapes):
        """Test basic token shift operation."""
        x = jax.random.normal(key, (
            small_shapes["batch"],
            small_shapes["seq_len"],
            small_shapes["hidden_size"],
        ))
        
        delta, new_shift = token_shift(x)
        
        assert delta.shape == x.shape
        assert new_shift.shape == (small_shapes["batch"], small_shapes["hidden_size"])
        
        # First position should be -x (shifted is zero-padded)
        assert jnp.allclose(delta[:, 0, :], -x[:, 0, :])
        
        # New shift should be last token
        assert jnp.allclose(new_shift, x[:, -1, :])

    def test_token_shift_with_state(self, key, small_shapes):
        """Test token shift with previous shift state."""
        x = jax.random.normal(key, (
            small_shapes["batch"],
            small_shapes["seq_len"],
            small_shapes["hidden_size"],
        ))
        prev_shift = jax.random.normal(
            jax.random.split(key)[1],
            (small_shapes["batch"], small_shapes["hidden_size"])
        )
        
        delta, new_shift = token_shift(x, shift_state=prev_shift)
        
        # First position should be prev_shift - x[0]
        assert jnp.allclose(delta[:, 0, :], prev_shift - x[:, 0, :])

    def test_token_shift_single_token(self, key, small_shapes):
        """Test token shift for single token (generation mode)."""
        x = jax.random.normal(key, (
            small_shapes["batch"],
            1,
            small_shapes["hidden_size"],
        ))
        prev_shift = jax.random.normal(
            jax.random.split(key)[1],
            (small_shapes["batch"], small_shapes["hidden_size"])
        )
        
        delta, new_shift = token_shift(x, shift_state=prev_shift)
        
        assert delta.shape == (small_shapes["batch"], 1, small_shapes["hidden_size"])
        assert new_shift.shape == (small_shapes["batch"], small_shapes["hidden_size"])
        assert jnp.allclose(new_shift, x[:, 0, :])


# =============================================================================
# Kernel Tests
# =============================================================================

class TestRWKV6Kernels:
    """Tests for RWKV6 kernel functions."""

    def test_recurrent_output_shape(self, key, small_shapes):
        """Test output shapes from recurrent kernel."""
        keys = jax.random.split(key, 5)
        batch = small_shapes["batch"]
        num_heads = small_shapes["num_heads"]
        seq_len = small_shapes["seq_len"]
        head_dim = small_shapes["head_dim"]
        
        r = jax.random.normal(keys[0], (batch, num_heads, seq_len, head_dim))
        k = jax.random.normal(keys[1], (batch, num_heads, seq_len, head_dim))
        v = jax.random.normal(keys[2], (batch, num_heads, seq_len, head_dim))
        w = -jnp.abs(jax.random.normal(keys[3], (batch, num_heads, seq_len, head_dim)))
        u = jax.random.normal(keys[4], (num_heads, head_dim))
        
        output, state = rwkv6_recurrent(r, k, v, w, u)
        
        assert output.shape == (batch, num_heads, seq_len, head_dim)
        assert state.shape == (batch, num_heads, head_dim, head_dim)

    def test_recurrent_with_initial_state(self, key, small_shapes):
        """Test recurrent kernel with initial state."""
        keys = jax.random.split(key, 6)
        batch = small_shapes["batch"]
        num_heads = small_shapes["num_heads"]
        seq_len = small_shapes["seq_len"]
        head_dim = small_shapes["head_dim"]
        
        r = jax.random.normal(keys[0], (batch, num_heads, seq_len, head_dim))
        k = jax.random.normal(keys[1], (batch, num_heads, seq_len, head_dim))
        v = jax.random.normal(keys[2], (batch, num_heads, seq_len, head_dim))
        w = -jnp.abs(jax.random.normal(keys[3], (batch, num_heads, seq_len, head_dim)))
        u = jax.random.normal(keys[4], (num_heads, head_dim))
        initial_state = jax.random.normal(keys[5], (batch, num_heads, head_dim, head_dim))
        
        output, state = rwkv6_recurrent(r, k, v, w, u, initial_state=initial_state)
        
        assert output.shape == (batch, num_heads, seq_len, head_dim)
        assert state.shape == (batch, num_heads, head_dim, head_dim)
        # Output should be different with initial state
        output_no_state, _ = rwkv6_recurrent(r, k, v, w, u, initial_state=None)
        assert not jnp.allclose(output, output_no_state, rtol=1e-4)

    def test_step_output_shape(self, key, small_shapes):
        """Test single step output shape."""
        keys = jax.random.split(key, 5)
        batch = small_shapes["batch"]
        num_heads = small_shapes["num_heads"]
        head_dim = small_shapes["head_dim"]
        
        r = jax.random.normal(keys[0], (batch, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch, num_heads, head_dim))
        w = -jnp.abs(jax.random.normal(keys[3], (batch, num_heads, head_dim)))
        u = jax.random.normal(keys[4], (num_heads, head_dim))
        state = jnp.zeros((batch, num_heads, head_dim, head_dim))
        
        output, new_state = rwkv6_step(r, k, v, w, u, state)
        
        assert output.shape == (batch, num_heads, head_dim)
        assert new_state.shape == (batch, num_heads, head_dim, head_dim)

    def test_step_matches_first_recurrent_step(self, key, small_shapes):
        """Single step should match first step of recurrent."""
        keys = jax.random.split(key, 5)
        batch = small_shapes["batch"]
        num_heads = small_shapes["num_heads"]
        head_dim = small_shapes["head_dim"]
        
        r = jax.random.normal(keys[0], (batch, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch, num_heads, head_dim))
        w = -jnp.abs(jax.random.normal(keys[3], (batch, num_heads, head_dim)))
        u = jax.random.normal(keys[4], (num_heads, head_dim))
        
        # Single step
        out_step, state_step = rwkv6_step(r, k, v, w, u, jnp.zeros((batch, num_heads, head_dim, head_dim)))
        
        # Recurrent with seq_len=1
        r_seq = r[:, :, None, :]
        k_seq = k[:, :, None, :]
        v_seq = v[:, :, None, :]
        w_seq = w[:, :, None, :]
        out_rec, state_rec = rwkv6_recurrent(r_seq, k_seq, v_seq, w_seq, u)
        
        assert jnp.allclose(out_step, out_rec[:, :, 0, :], rtol=1e-4)
        assert jnp.allclose(state_step, state_rec, rtol=1e-4)

    def test_outputs_are_finite(self, key, small_shapes):
        """Recurrent should produce finite outputs."""
        keys = jax.random.split(key, 5)
        batch = small_shapes["batch"]
        num_heads = small_shapes["num_heads"]
        seq_len = small_shapes["seq_len"]
        head_dim = small_shapes["head_dim"]
        
        r = jax.random.normal(keys[0], (batch, num_heads, seq_len, head_dim))
        k = jax.random.normal(keys[1], (batch, num_heads, seq_len, head_dim))
        v = jax.random.normal(keys[2], (batch, num_heads, seq_len, head_dim))
        w = -jnp.abs(jax.random.normal(keys[3], (batch, num_heads, seq_len, head_dim)))
        u = jax.random.normal(keys[4], (num_heads, head_dim))
        
        output, state = rwkv6_recurrent(r, k, v, w, u)
        
        assert jnp.all(jnp.isfinite(output))
        assert jnp.all(jnp.isfinite(state))

    def test_decay_behavior(self, key, small_shapes):
        """Test that decay (w) properly affects state."""
        keys = jax.random.split(key, 5)
        batch = small_shapes["batch"]
        num_heads = small_shapes["num_heads"]
        seq_len = 4  # Small sequence for clarity
        head_dim = small_shapes["head_dim"]
        
        r = jax.random.normal(keys[0], (batch, num_heads, seq_len, head_dim))
        k = jax.random.normal(keys[1], (batch, num_heads, seq_len, head_dim))
        v = jax.random.normal(keys[2], (batch, num_heads, seq_len, head_dim))
        u = jax.random.normal(keys[4], (num_heads, head_dim))
        
        # RWKV-6 uses decay = exp(-exp(w)). More-negative w => exp(w) smaller => decay closer to 1.
        # So w=-10 corresponds to *weaker* decay than w=-0.1.
        w_weaker_decay = -10.0 * jnp.ones((batch, num_heads, seq_len, head_dim))
        _, state_weaker = rwkv6_recurrent(r, k, v, w_weaker_decay, u)

        w_stronger_decay = -0.1 * jnp.ones((batch, num_heads, seq_len, head_dim))
        _, state_stronger = rwkv6_recurrent(r, k, v, w_stronger_decay, u)

        # Weaker decay should accumulate more state
        state_weaker_norm = jnp.linalg.norm(state_weaker)
        state_stronger_norm = jnp.linalg.norm(state_stronger)
        assert state_weaker_norm > state_stronger_norm


# =============================================================================
# Block Tests
# =============================================================================

class TestRWKV6Block:
    """Tests for RWKV6Block module."""

    def test_forward_shape(self, key, small_shapes):
        """Test forward pass output shape."""
        block = RWKV6Block(
            hidden_size=small_shapes["hidden_size"],
            num_heads=small_shapes["num_heads"],
            intermediate_size=small_shapes["hidden_size"] * 4,
            layer_idx=0,
            n_layers=12,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (
            small_shapes["batch"],
            small_shapes["seq_len"],
            small_shapes["hidden_size"],
        ))
        
        output, state = block(x)
        
        assert output.shape == x.shape

    def test_forward_with_state(self, key, small_shapes):
        """Test forward pass with state caching."""
        block = RWKV6Block(
            hidden_size=small_shapes["hidden_size"],
            num_heads=small_shapes["num_heads"],
            intermediate_size=small_shapes["hidden_size"] * 4,
            layer_idx=0,
            n_layers=12,
            rngs=nnx.Rngs(0),
        )
        
        x = jax.random.normal(key, (
            small_shapes["batch"],
            small_shapes["seq_len"],
            small_shapes["hidden_size"],
        ))
        
        # First pass - no state
        output1, state1 = block(x, mode="recurrent")
        
        assert state1 is not None
        assert isinstance(state1, RWKV6State)
        assert state1.h is not None
        assert state1.shift_state is not None

    def test_sequential_generation(self, key, small_shapes):
        """Test multi-step autoregressive generation."""
        block = RWKV6Block(
            hidden_size=small_shapes["hidden_size"],
            num_heads=small_shapes["num_heads"],
            intermediate_size=small_shapes["hidden_size"] * 4,
            layer_idx=0,
            n_layers=12,
            rngs=nnx.Rngs(0),
        )
        
        batch = small_shapes["batch"]
        hidden = small_shapes["hidden_size"]
        
        state = block.init_state(batch)
        
        for i in range(5):
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (batch, 1, hidden))
            output, state = block(x, state=state, mode="recurrent")
            
            assert output.shape == (batch, 1, hidden)
            assert state is not None

    def test_init_state(self, small_shapes):
        """Test state initialization."""
        block = RWKV6Block(
            hidden_size=small_shapes["hidden_size"],
            num_heads=small_shapes["num_heads"],
            intermediate_size=small_shapes["hidden_size"] * 4,
            layer_idx=0,
            n_layers=12,
            rngs=nnx.Rngs(0),
        )
        
        state = block.init_state(small_shapes["batch"])
        
        assert isinstance(state, RWKV6State)
        assert state.h.shape == (
            small_shapes["batch"],
            small_shapes["num_heads"],
            small_shapes["hidden_size"] // small_shapes["num_heads"],
            small_shapes["hidden_size"] // small_shapes["num_heads"],
        )

    def test_different_layer_idx(self, key, small_shapes):
        """Test that different layer_idx produces different initializations."""
        block0 = RWKV6Block(
            hidden_size=small_shapes["hidden_size"],
            num_heads=small_shapes["num_heads"],
            layer_idx=0,
            n_layers=12,
            rngs=nnx.Rngs(0),
        )
        
        block11 = RWKV6Block(
            hidden_size=small_shapes["hidden_size"],
            num_heads=small_shapes["num_heads"],
            layer_idx=11,
            n_layers=12,
            rngs=nnx.Rngs(0),
        )
        
        # The time_maa_x parameters should be different
        assert not jnp.allclose(block0.time_maa_x.value, block11.time_maa_x.value)


# =============================================================================
# Model Integration Tests
# =============================================================================

class TestModelIntegration:
    """Tests for RWKV6 integration with LMModel."""

    def test_rwkv6_model_forward(self, key):
        """Test RWKV6 in full model."""
        from linearnexus.models import ModelConfig, LMModel
        
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            n_layers=2,
            n_heads=2,
            block_pattern=["rwkv6"],
            rwkv6_heads=2,
            rwkv6_intermediate_size=256,
        )
        
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        tokens = jax.random.randint(key, (2, 16), 0, 100)
        logits, state = model(tokens)
        
        assert logits.shape == (2, 16, 100)
        assert jnp.all(jnp.isfinite(logits))

    def test_rwkv6_hybrid_model(self, key):
        """Test RWKV6 + Attention hybrid."""
        from linearnexus.models import ModelConfig, LMModel
        
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            n_layers=4,
            n_heads=2,
            block_pattern=["rwkv6", "rwkv6", "attention"],
            rwkv6_heads=2,
        )
        
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        tokens = jax.random.randint(key, (2, 16), 0, 100)
        logits, state = model(tokens)
        
        assert logits.shape == (2, 16, 100)
        assert jnp.all(jnp.isfinite(logits))

    def test_create_model_preset(self):
        """Test create_model with RWKV6 preset."""
        from linearnexus.models import create_model
        
        # Test with compatible overrides (keep hidden_size and heads consistent)
        model = create_model("rwkv6-small", rngs=nnx.Rngs(0), n_layers=2)
        
        assert model is not None

    def test_model_with_generation(self, key):
        """Test model generation with state caching."""
        from linearnexus.models import ModelConfig, LMModel
        
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            n_layers=2,
            n_heads=2,
            block_pattern=["rwkv6"],
            rwkv6_heads=2,
        )
        
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        # Initial forward
        tokens = jax.random.randint(key, (2, 4), 0, 100)
        logits, state = model(tokens, mode="recurrent")
        
        # Generation step
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        logits2, state2 = model(next_token, state=state, mode="recurrent")
        
        assert logits2.shape == (2, 1, 100)
        assert jnp.all(jnp.isfinite(logits2))


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability."""

    def test_long_sequence(self, key, small_shapes):
        """Test with longer sequences."""
        block = RWKV6Block(
            hidden_size=small_shapes["hidden_size"],
            num_heads=small_shapes["num_heads"],
            layer_idx=0,
            n_layers=12,
            rngs=nnx.Rngs(0),
        )
        
        # Longer sequence
        x = jax.random.normal(key, (
            small_shapes["batch"],
            256,  # Longer sequence
            small_shapes["hidden_size"],
        ))
        
        # Use recurrent mode to get state back
        output, state = block(x, mode="recurrent")
        
        assert jnp.all(jnp.isfinite(output))
        assert state is not None
        assert jnp.all(jnp.isfinite(state.h))

    def test_many_generation_steps(self, key, small_shapes):
        """Test many sequential generation steps."""
        block = RWKV6Block(
            hidden_size=small_shapes["hidden_size"],
            num_heads=small_shapes["num_heads"],
            layer_idx=0,
            n_layers=12,
            rngs=nnx.Rngs(0),
        )
        
        state = block.init_state(small_shapes["batch"])
        
        for i in range(100):
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (
                small_shapes["batch"],
                1,
                small_shapes["hidden_size"],
            ))
            output, state = block(x, state=state, mode="recurrent")
            
            assert jnp.all(jnp.isfinite(output)), f"NaN/Inf at step {i}"
            assert jnp.all(jnp.isfinite(state.h)), f"State NaN/Inf at step {i}"
