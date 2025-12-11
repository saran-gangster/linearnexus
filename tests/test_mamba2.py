"""Tests for Mamba2 implementation.

Validates:
- Shape correctness through the block
- State caching for generation
- Chunk vs recurrent mode parity
- SSD algorithm correctness
"""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.modules.ssm.mamba2 import (
    Mamba2Block,
    Mamba2State,
    RMSNormGated,
    segment_sum,
    ssd_chunk_scan,
    ssd_recurrent_step,
)


class TestRMSNormGated:
    """Tests for gated RMSNorm."""
    
    def test_shape_preservation(self):
        """Output shape should match input."""
        rngs = nnx.Rngs(0)
        norm = RMSNormGated(64, rngs=rngs)
        
        x = jnp.ones((2, 8, 64))
        gate = jnp.ones((2, 8, 64))
        
        out = norm(x, gate)
        assert out.shape == x.shape
    
    def test_gating_effect(self):
        """Zero gate should produce zero output."""
        rngs = nnx.Rngs(0)
        norm = RMSNormGated(64, rngs=rngs)
        
        x = jnp.ones((2, 8, 64))
        gate = jnp.zeros((2, 8, 64))  # Zero gate
        
        out = norm(x, gate)
        assert jnp.allclose(out, 0.0, atol=1e-6)


class TestSegmentSum:
    """Tests for segment_sum helper."""
    
    def test_output_shape(self):
        """Should produce [*, chunk, chunk] from [*, chunk]."""
        x = jnp.ones((2, 4, 8))  # batch, heads, chunk
        out = segment_sum(x)
        assert out.shape == (2, 4, 8, 8)
    
    def test_lower_triangular_structure(self):
        """Result should have -inf in strict upper triangle."""
        x = jnp.ones((4,))
        out = segment_sum(x)
        
        # Check strict upper triangle (k=1) is -inf
        upper_strict = jnp.triu(out, k=1)
        # Upper strict triangle should be -inf
        assert jnp.all(jnp.isinf(upper_strict) | (upper_strict == 0))


class TestMamba2State:
    """Tests for Mamba2State dataclass."""
    
    def test_zeros_creation(self):
        """Should create correctly shaped zero state."""
        state = Mamba2State.zeros(
            batch_size=2,
            num_heads=4,
            head_dim=32,
            conv_kernel=4,
            conv_dim=128 + 2 * 1 * 64,  # intermediate + 2*groups*state
            state_size=64,
        )
        
        assert state.conv_state.shape == (2, 3, 128 + 128)  # kernel-1
        assert state.ssm_state.shape == (2, 4, 32, 64)
        assert state.position == 0


class TestSSDChunkScan:
    """Tests for SSD chunk scan algorithm."""
    
    def test_output_shape(self):
        """Output should match input spatial dims."""
        batch, seq, heads, dim = 2, 16, 4, 32
        state_size = 64
        n_groups = 1
        
        hidden = jnp.ones((batch, seq, heads, dim))
        dt = jnp.ones((batch, seq, heads)) * 0.1
        A = -jnp.ones((heads,))
        B = jnp.ones((batch, seq, n_groups, state_size))
        C = jnp.ones((batch, seq, n_groups, state_size))
        D = jnp.ones((heads,))
        
        out, state = ssd_chunk_scan(
            hidden, dt, A, B, C, D,
            chunk_size=8,
            n_groups=n_groups,
        )
        
        assert out.shape == (batch, seq, heads, dim)
        assert state.shape == (batch, heads, dim, state_size)
    
    def test_with_initial_state(self):
        """Should accept and use initial state."""
        batch, seq, heads, dim = 2, 8, 4, 32
        state_size = 64
        
        hidden = jnp.ones((batch, seq, heads, dim))
        dt = jnp.ones((batch, seq, heads)) * 0.1
        A = -jnp.ones((heads,))
        B = jnp.ones((batch, seq, 1, state_size))
        C = jnp.ones((batch, seq, 1, state_size))
        D = jnp.ones((heads,))
        
        init_state = jnp.ones((batch, heads, dim, state_size)) * 0.5
        
        out, final_state = ssd_chunk_scan(
            hidden, dt, A, B, C, D,
            chunk_size=8,
            n_groups=1,
            ssm_state=init_state,
        )
        
        # State should be updated (not equal to init)
        assert not jnp.allclose(final_state, init_state)


class TestSSDRecurrentStep:
    """Tests for single-step recurrent update."""
    
    def test_output_shape(self):
        """Output should be [batch, heads, dim]."""
        batch, heads, dim = 2, 4, 32
        state_size = 64
        n_groups = 1
        
        hidden = jnp.ones((batch, heads, dim))
        dt = jnp.ones((batch, heads)) * 0.1
        A = -jnp.ones((heads,))
        B = jnp.ones((batch, n_groups, state_size))
        C = jnp.ones((batch, n_groups, state_size))
        D = jnp.ones((heads,))
        ssm_state = jnp.zeros((batch, heads, dim, state_size))
        
        out, new_state = ssd_recurrent_step(
            hidden, dt, A, B, C, D, ssm_state,
            n_groups=n_groups,
        )
        
        assert out.shape == (batch, heads, dim)
        assert new_state.shape == (batch, heads, dim, state_size)


class TestMamba2Block:
    """Tests for the full Mamba2Block module."""
    
    @pytest.fixture
    def block(self):
        """Create a small Mamba2 block for testing."""
        return Mamba2Block(
            hidden_size=64,
            num_heads=4,
            head_dim=16,
            state_size=32,
            n_groups=1,
            conv_kernel=4,
            chunk_size=8,
            rngs=nnx.Rngs(0),
        )
    
    def test_output_shape(self, block):
        """Output should match input shape."""
        x = jnp.ones((2, 16, 64))
        out, state = block(x)
        
        assert out.shape == x.shape
        assert isinstance(state, Mamba2State)
    
    def test_state_initialization(self, block):
        """init_state should create valid state."""
        state = block.init_state(batch_size=2)
        
        assert isinstance(state, Mamba2State)
        assert state.position == 0
    
    def test_chunk_mode(self, block):
        """Should work in chunk mode."""
        x = jnp.ones((2, 16, 64))
        out, state = block(x, mode="chunk")
        
        assert out.shape == x.shape
    
    def test_recurrent_mode(self, block):
        """Should work in recurrent mode."""
        x = jnp.ones((2, 1, 64))  # Single token
        out, state = block(x, mode="recurrent")
        
        assert out.shape == x.shape
    
    def test_state_continuation(self, block):
        """State should allow continuation of sequence."""
        x1 = jnp.ones((2, 8, 64))
        x2 = jnp.ones((2, 8, 64))
        
        # First pass
        out1, state1 = block(x1)
        
        # Continue with state
        out2, state2 = block(x2, state=state1)
        
        assert state2.position == 16
        assert out2.shape == x2.shape
    
    def test_autoregressive_generation(self, block):
        """Should support token-by-token generation."""
        batch_size = 2
        
        # Initial state
        state = block.init_state(batch_size)
        
        outputs = []
        for _ in range(8):
            x = jnp.ones((batch_size, 1, 64))
            out, state = block(x, state=state, mode="recurrent")
            outputs.append(out)
        
        # Check we got outputs for all tokens
        assert len(outputs) == 8
        assert state.position == 8
    
    def test_deterministic(self, block):
        """Same input should produce same output."""
        x = jnp.ones((2, 8, 64))
        
        out1, _ = block(x)
        out2, _ = block(x)
        
        assert jnp.allclose(out1, out2)


class TestMamba2ChunkRecurrentParity:
    """Test that chunk and recurrent modes produce similar results."""
    
    def test_chunk_vs_recurrent_parity(self):
        """Chunk and sequential recurrent should be close."""
        rngs = nnx.Rngs(42)
        block = Mamba2Block(
            hidden_size=64,
            num_heads=4,
            head_dim=16,
            state_size=32,
            n_groups=1,
            conv_kernel=4,
            chunk_size=8,
            rngs=rngs,
        )
        
        # Random input
        key = jax.random.PRNGKey(123)
        x = jax.random.normal(key, (1, 8, 64))
        
        # Chunk mode (full sequence)
        out_chunk, state_chunk = block(x, mode="chunk")
        
        # Recurrent mode (token by token)
        state_rec = block.init_state(1)
        outputs_rec = []
        for t in range(8):
            out_t, state_rec = block(x[:, t:t+1, :], state=state_rec, mode="recurrent")
            outputs_rec.append(out_t)
        out_rec = jnp.concatenate(outputs_rec, axis=1)
        
        # Should be close (not exact due to numerical differences)
        assert jnp.allclose(out_chunk, out_rec, rtol=1e-3, atol=1e-3), \
            f"Max diff: {jnp.abs(out_chunk - out_rec).max()}"


class TestMamba2Integration:
    """Integration tests with LMModel."""
    
    def test_create_mamba2_model(self):
        """Should be able to create a Mamba2-based model."""
        from linearnexus import create_model, LMModel
        
        config, _ = create_model("mamba2-small", vocab_size=256)
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        # Forward pass
        tokens = jnp.ones((2, 16), dtype=jnp.int32)
        logits, state = model(tokens)
        
        assert logits.shape == (2, 16, 256)
    
    def test_mamba2_generation(self):
        """Should support autoregressive generation."""
        from linearnexus import create_model, LMModel
        
        config, _ = create_model("mamba2-small", vocab_size=256, n_layers=2)
        model = LMModel(config, rngs=nnx.Rngs(0))
        
        # Initial state
        state = model.init_state(batch_size=1)
        
        # Generate tokens
        for _ in range(4):
            token = jnp.ones((1, 1), dtype=jnp.int32)
            logits, state = model(token, state=state, mode="recurrent")
        
        assert logits.shape == (1, 1, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
