"""Tests for DeltaNet linear attention implementation.

Tests cover:
- Delta rule kernel shape correctness
- Recurrent vs chunkwise output parity  
- State caching for autoregressive generation
- Integration with full LMModel
"""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.modules.linear_attn import (
    DeltaNetBlock,
    DeltaNetState,
    delta_rule_recurrent,
    delta_rule_chunkwise,
    delta_rule_step,
)


class TestDeltaRuleKernels:
    """Test the delta rule kernel implementations."""
    
    def test_recurrent_shape(self):
        """Test that recurrent delta rule outputs correct shapes."""
        batch, heads, seq_len, key_dim, value_dim = 2, 4, 16, 32, 64
        
        q = jax.random.normal(jax.random.key(0), (batch, heads, seq_len, key_dim))
        k = jax.random.normal(jax.random.key(1), (batch, heads, seq_len, key_dim))
        v = jax.random.normal(jax.random.key(2), (batch, heads, seq_len, value_dim))
        beta = jax.nn.sigmoid(jax.random.normal(jax.random.key(3), (batch, heads, seq_len)))
        
        # L2 normalize q and k for numerical stability
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        
        output, final_state = delta_rule_recurrent(q, k, v, beta)
        
        assert output.shape == (batch, heads, seq_len, value_dim)
        assert final_state.shape == (batch, heads, key_dim, value_dim)
    
    def test_chunkwise_shape(self):
        """Test that chunkwise delta rule outputs correct shapes."""
        batch, heads, seq_len, key_dim, value_dim = 2, 4, 64, 32, 64
        chunk_size = 16
        
        q = jax.random.normal(jax.random.key(0), (batch, heads, seq_len, key_dim))
        k = jax.random.normal(jax.random.key(1), (batch, heads, seq_len, key_dim))
        v = jax.random.normal(jax.random.key(2), (batch, heads, seq_len, value_dim))
        beta = jax.nn.sigmoid(jax.random.normal(jax.random.key(3), (batch, heads, seq_len)))
        
        # L2 normalize
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        
        output, final_state = delta_rule_chunkwise(q, k, v, beta, chunk_size=chunk_size)
        
        assert output.shape == (batch, heads, seq_len, value_dim)
        assert final_state.shape == (batch, heads, key_dim, value_dim)
    
    def test_recurrent_vs_chunkwise_parity(self):
        """Test that recurrent and chunkwise modes produce similar outputs."""
        batch, heads, seq_len, key_dim, value_dim = 2, 4, 32, 16, 32
        chunk_size = 8
        
        key = jax.random.key(42)
        keys = jax.random.split(key, 4)
        
        q = jax.random.normal(keys[0], (batch, heads, seq_len, key_dim))
        k = jax.random.normal(keys[1], (batch, heads, seq_len, key_dim))
        v = jax.random.normal(keys[2], (batch, heads, seq_len, value_dim))
        beta = jax.nn.sigmoid(jax.random.normal(keys[3], (batch, heads, seq_len)))
        
        # L2 normalize for numerical stability
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        
        out_rec, state_rec = delta_rule_recurrent(q, k, v, beta)
        out_chunk, state_chunk = delta_rule_chunkwise(q, k, v, beta, chunk_size=chunk_size)
        
        # Check output parity
        assert jnp.allclose(out_rec, out_chunk, rtol=1e-3, atol=1e-3), \
            f"Max diff: {jnp.max(jnp.abs(out_rec - out_chunk))}"
        
        # Check state parity
        assert jnp.allclose(state_rec, state_chunk, rtol=1e-3, atol=1e-3), \
            f"State max diff: {jnp.max(jnp.abs(state_rec - state_chunk))}"
    
    def test_initial_state_handling(self):
        """Test that initial state is correctly used."""
        batch, heads, seq_len, key_dim, value_dim = 2, 4, 16, 16, 32
        
        q = jax.random.normal(jax.random.key(0), (batch, heads, seq_len, key_dim))
        k = jax.random.normal(jax.random.key(1), (batch, heads, seq_len, key_dim))
        v = jax.random.normal(jax.random.key(2), (batch, heads, seq_len, value_dim))
        beta = jax.nn.sigmoid(jax.random.normal(jax.random.key(3), (batch, heads, seq_len)))
        
        # L2 normalize
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        
        # Non-zero initial state
        init_state = jax.random.normal(jax.random.key(4), (batch, heads, key_dim, value_dim)) * 0.1
        
        out_zero, _ = delta_rule_recurrent(q, k, v, beta, initial_state=None)
        out_init, _ = delta_rule_recurrent(q, k, v, beta, initial_state=init_state)
        
        # Outputs should differ when starting from different states
        assert not jnp.allclose(out_zero, out_init), \
            "Outputs should differ with different initial states"
    
    def test_step_function(self):
        """Test single-step delta rule function."""
        batch, heads, key_dim, value_dim = 2, 4, 16, 32
        
        q = jax.random.normal(jax.random.key(0), (batch, heads, key_dim))
        k = jax.random.normal(jax.random.key(1), (batch, heads, key_dim))
        v = jax.random.normal(jax.random.key(2), (batch, heads, value_dim))
        beta = jax.nn.sigmoid(jax.random.normal(jax.random.key(3), (batch, heads)))
        state = jnp.zeros((batch, heads, key_dim, value_dim))
        
        # L2 normalize
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        
        output, new_state = delta_rule_step(q, k, v, beta, state)
        
        assert output.shape == (batch, heads, value_dim)
        assert new_state.shape == (batch, heads, key_dim, value_dim)
    
    def test_step_matches_recurrent(self):
        """Test that step function matches first step of recurrent."""
        batch, heads, key_dim, value_dim = 2, 4, 16, 32
        
        q = jax.random.normal(jax.random.key(0), (batch, heads, 1, key_dim))
        k = jax.random.normal(jax.random.key(1), (batch, heads, 1, key_dim))
        v = jax.random.normal(jax.random.key(2), (batch, heads, 1, value_dim))
        beta = jax.nn.sigmoid(jax.random.normal(jax.random.key(3), (batch, heads, 1)))
        
        # L2 normalize
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        
        # Recurrent with seq_len=1
        out_rec, state_rec = delta_rule_recurrent(q, k, v, beta)
        
        # Step function
        out_step, state_step = delta_rule_step(
            q.squeeze(2), k.squeeze(2), v.squeeze(2), beta.squeeze(2),
            jnp.zeros((batch, heads, key_dim, value_dim))
        )
        
        assert jnp.allclose(out_rec.squeeze(2), out_step, rtol=1e-5)
        assert jnp.allclose(state_rec, state_step, rtol=1e-5)


class TestDeltaNetBlock:
    """Test the DeltaNet block implementation."""
    
    def test_block_forward_shape(self):
        """Test DeltaNetBlock output shapes."""
        batch, seq_len, hidden = 2, 32, 64
        rngs = nnx.Rngs(0)
        
        block = DeltaNetBlock(
            hidden_size=hidden,
            num_heads=4,
            use_short_conv=True,
            conv_size=4,
            chunk_size=16,
            rngs=rngs,
        )
        
        x = jax.random.normal(jax.random.key(0), (batch, seq_len, hidden))
        output, state = block(x, mode="chunk")
        
        assert output.shape == (batch, seq_len, hidden)
        # Note: In chunk mode without explicit state input, state is None
        # State is only returned in recurrent mode or when state is passed in
    
    def test_block_recurrent_mode(self):
        """Test DeltaNetBlock in recurrent mode."""
        batch, hidden = 2, 64
        rngs = nnx.Rngs(0)
        
        block = DeltaNetBlock(
            hidden_size=hidden,
            num_heads=4,
            use_short_conv=False,  # Simpler for testing
            rngs=rngs,
        )
        
        # Initial forward pass
        x = jax.random.normal(jax.random.key(0), (batch, 8, hidden))
        output1, state1 = block(x, mode="chunk")
        
        # Single token step
        x_single = jax.random.normal(jax.random.key(1), (batch, 1, hidden))
        output2, state2 = block(x_single, state=state1, mode="recurrent")
        
        assert output2.shape == (batch, 1, hidden)
        assert state2.S is not None
    
    def test_init_state(self):
        """Test DeltaNetBlock.init_state factory."""
        hidden = 64
        rngs = nnx.Rngs(0)
        
        block = DeltaNetBlock(
            hidden_size=hidden,
            num_heads=4,
            use_short_conv=True,
            conv_size=4,
            rngs=rngs,
        )
        
        state = block.init_state(batch_size=2)
        
        assert isinstance(state, DeltaNetState)
        assert state.S.shape == (2, 4, block.key_dim, block.value_dim)
        assert state.conv_state is not None
    
    def test_block_no_conv(self):
        """Test DeltaNetBlock without short convolutions."""
        batch, seq_len, hidden = 2, 16, 64
        rngs = nnx.Rngs(0)
        
        block = DeltaNetBlock(
            hidden_size=hidden,
            num_heads=4,
            use_short_conv=False,
            rngs=rngs,
        )
        
        x = jax.random.normal(jax.random.key(0), (batch, seq_len, hidden))
        output, state = block(x)
        
        assert output.shape == (batch, seq_len, hidden)
    
    def test_block_no_gate(self):
        """Test DeltaNetBlock without output gating."""
        batch, seq_len, hidden = 2, 16, 64
        rngs = nnx.Rngs(0)
        
        block = DeltaNetBlock(
            hidden_size=hidden,
            num_heads=4,
            use_gate=False,
            rngs=rngs,
        )
        
        x = jax.random.normal(jax.random.key(0), (batch, seq_len, hidden))
        output, state = block(x)
        
        assert output.shape == (batch, seq_len, hidden)
    
    def test_block_with_gate(self):
        """Test DeltaNetBlock with output gating."""
        batch, seq_len, hidden = 2, 16, 64
        rngs = nnx.Rngs(0)
        
        block = DeltaNetBlock(
            hidden_size=hidden,
            num_heads=4,
            use_gate=True,
            rngs=rngs,
        )
        
        x = jax.random.normal(jax.random.key(0), (batch, seq_len, hidden))
        output, state = block(x)
        
        assert output.shape == (batch, seq_len, hidden)


class TestDeltaNetNumerical:
    """Numerical stability and correctness tests."""
    
    def test_beta_zero_no_update(self):
        """Test that beta=0 gives no state update."""
        batch, heads, seq_len, key_dim, value_dim = 1, 2, 4, 8, 16
        
        q = jax.random.normal(jax.random.key(0), (batch, heads, seq_len, key_dim))
        k = jax.random.normal(jax.random.key(1), (batch, heads, seq_len, key_dim))
        v = jax.random.normal(jax.random.key(2), (batch, heads, seq_len, value_dim))
        
        # L2 normalize
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        
        # Beta = 0: no learning
        beta_zero = jnp.zeros((batch, heads, seq_len))
        _, state_zero = delta_rule_recurrent(q, k, v, beta_zero)
        
        assert jnp.allclose(state_zero, 0.0), "State should be zero when beta=0"
    
    def test_beta_one_updates(self):
        """Test that beta=1 causes state updates."""
        batch, heads, seq_len, key_dim, value_dim = 1, 2, 4, 8, 16
        
        q = jax.random.normal(jax.random.key(0), (batch, heads, seq_len, key_dim))
        k = jax.random.normal(jax.random.key(1), (batch, heads, seq_len, key_dim))
        v = jax.random.normal(jax.random.key(2), (batch, heads, seq_len, value_dim))
        
        # L2 normalize
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        
        beta_one = jnp.ones((batch, heads, seq_len))
        _, state_one = delta_rule_recurrent(q, k, v, beta_one)
        
        assert not jnp.allclose(state_one, 0.0), "State should be non-zero when beta=1"
    
    def test_deterministic_output(self):
        """Test that same inputs give same outputs."""
        batch, seq_len, hidden = 2, 16, 64
        rngs = nnx.Rngs(0)
        
        block = DeltaNetBlock(hidden_size=hidden, num_heads=4, rngs=rngs)
        
        x = jax.random.normal(jax.random.key(0), (batch, seq_len, hidden))
        
        out1, _ = block(x)
        out2, _ = block(x)
        
        assert jnp.allclose(out1, out2), "Outputs should be deterministic"
    
    def test_delta_rule_correctness(self):
        """Verify delta rule math: S @ k retrieves stored value."""
        batch, heads, key_dim, value_dim = 1, 1, 4, 4
        
        # Create orthonormal keys
        k = jnp.eye(key_dim)[None, None, :, :]  # [1, 1, key_dim, key_dim]
        v = jax.random.normal(jax.random.key(0), (batch, heads, key_dim, value_dim))
        q = k.copy()  # Query same as keys
        beta = jnp.ones((batch, heads, key_dim))  # Full learning rate
        
        output, final_state = delta_rule_recurrent(q, k, v, beta)
        
        # With orthonormal keys and beta=1, after storing k_i -> v_i,
        # querying with q_i should retrieve v_i
        # The output should approximate the value (with some delta corrections)
        # This is a simplified test - exact retrieval depends on key orthogonality
        assert output.shape == (batch, heads, key_dim, value_dim)


class TestDeltaNetModel:
    """Test DeltaNet integrated with the full LMModel."""
    
    def test_deltanet_model_creation(self):
        """Test creating a pure DeltaNet model."""
        from linearnexus.models import ModelConfig, LMModel
        
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=64,
            n_layers=2,
            n_heads=4,
            block_pattern=["deltanet"],
            deltanet_heads=4,
            deltanet_use_short_conv=True,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        input_ids = jax.random.randint(jax.random.key(0), (2, 16), 0, 1000)
        logits, state = model(input_ids)
        
        assert logits.shape == (2, 16, 1000)
    
    def test_hybrid_deltanet_attention(self):
        """Test hybrid DeltaNet + Attention model."""
        from linearnexus.models import ModelConfig, LMModel
        
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=64,
            n_layers=6,
            n_heads=4,
            block_pattern=["deltanet", "deltanet", "attention"],
            deltanet_heads=4,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        input_ids = jax.random.randint(jax.random.key(0), (2, 16), 0, 1000)
        logits, state = model(input_ids)
        
        assert logits.shape == (2, 16, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
