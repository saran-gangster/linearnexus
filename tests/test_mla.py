"""Tests for Multi-Head Latent Attention (MLA) module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import flax.nnx as nnx

from linearnexus.modules.attention.mla import (
    MultiHeadLatentAttention,
    MLABlock,
    MLACache,
    apply_rope_mla,
)


class TestMLACache:
    """Tests for MLACache dataclass."""
    
    def test_zeros(self):
        """Test cache initialization."""
        batch_size, max_seq_len, kv_lora_rank, rope_head_dim = 2, 64, 128, 64
        cache = MLACache.zeros(batch_size, max_seq_len, kv_lora_rank, rope_head_dim)
        
        assert cache.compressed_kv.shape == (batch_size, max_seq_len, kv_lora_rank)
        assert cache.key_rope.shape == (batch_size, max_seq_len, 1, rope_head_dim)
        assert cache.position == 0
    
    def test_update_and_get(self):
        """Test cache update and retrieval."""
        batch_size, max_seq_len, kv_lora_rank, rope_head_dim = 2, 64, 128, 64
        cache = MLACache.zeros(batch_size, max_seq_len, kv_lora_rank, rope_head_dim)
        
        # Update with first chunk
        seq_len = 8
        compressed_kv = jnp.ones((batch_size, seq_len, kv_lora_rank))
        key_rope = jnp.ones((batch_size, seq_len, 1, rope_head_dim)) * 2.0
        
        cache = cache.update(compressed_kv, key_rope)
        assert cache.position == seq_len
        
        # Get cached values
        cached_kv, cached_rope = cache.get()
        assert cached_kv.shape == (batch_size, seq_len, kv_lora_rank)
        assert cached_rope.shape == (batch_size, seq_len, 1, rope_head_dim)
        np.testing.assert_array_almost_equal(cached_kv, jnp.ones_like(cached_kv))
        np.testing.assert_array_almost_equal(cached_rope, jnp.ones_like(cached_rope) * 2.0)


class TestMultiHeadLatentAttention:
    """Tests for MLA module."""
    
    @pytest.fixture
    def config(self):
        """Standard test configuration."""
        return dict(
            hidden_size=256,
            n_heads=8,
            v_head_dim=32,
            nope_head_dim=32,
            rope_head_dim=64,
            q_lora_rank=192,  # 3 * kv_lora_rank
            kv_lora_rank=64,
            max_seq_len=128,
        )
    
    def test_forward_shape(self, config):
        """Test that forward pass produces correct output shape."""
        rngs = nnx.Rngs(0)
        mla = MultiHeadLatentAttention(**config, rngs=rngs)
        
        batch_size, seq_len = 2, 16
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, config["hidden_size"]))
        
        output, state = mla(x)
        
        assert output.shape == x.shape
        assert state is None
    
    def test_forward_with_cache(self, config):
        """Test autoregressive generation with cache."""
        rngs = nnx.Rngs(0)
        mla = MultiHeadLatentAttention(**config, rngs=rngs)
        
        batch_size = 2
        
        # Initialize cache
        state = mla.init_state(batch_size, config["max_seq_len"])
        
        # Process initial sequence
        seq_len = 8
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, config["hidden_size"]))
        output1, state = mla(x, state=state)
        
        assert output1.shape == x.shape
        assert state.position == seq_len
        
        # Generate one token at a time
        x_next = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 1, config["hidden_size"]))
        output2, state = mla(x_next, state=state)
        
        assert output2.shape == x_next.shape
        assert state.position == seq_len + 1
    
    def test_causal_masking(self, config):
        """Test that attention is properly causal (no future leakage)."""
        rngs = nnx.Rngs(0)
        mla = MultiHeadLatentAttention(**config, rngs=rngs)
        
        batch_size, seq_len = 1, 8
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, config["hidden_size"]))
        
        # Get output for full sequence
        output_full, _ = mla(x)
        
        # Get output for truncated sequence (should match corresponding positions)
        output_partial, _ = mla(x[:, :4, :])
        
        # First 4 positions should be identical (causal = no future info)
        np.testing.assert_allclose(
            output_full[:, :4, :],
            output_partial,
            rtol=1e-5,
            err_msg="Causal masking violation: output depends on future tokens"
        )
    
    def test_parameter_count_reduction(self, config):
        """Verify MLA has fewer KV cache parameters than standard attention."""
        # MLA cache size: kv_lora_rank + rope_head_dim per position
        mla_cache_per_pos = config["kv_lora_rank"] + config["rope_head_dim"]
        
        # Standard MHA cache size: n_heads * head_dim * 2 (for K and V)
        # Assuming standard head_dim = hidden_size / n_heads
        standard_head_dim = config["hidden_size"] // config["n_heads"]
        standard_cache_per_pos = config["n_heads"] * standard_head_dim * 2
        
        print(f"MLA cache per position: {mla_cache_per_pos}")
        print(f"Standard cache per position: {standard_cache_per_pos}")
        print(f"Reduction: {standard_cache_per_pos / mla_cache_per_pos:.2f}x")
        
        # MLA should require significantly less cache
        assert mla_cache_per_pos < standard_cache_per_pos


class TestMLABlock:
    """Tests for MLABlock (attention + FFN)."""
    
    @pytest.fixture
    def config(self):
        """Standard test configuration."""
        return dict(
            hidden_size=256,
            n_heads=8,
            intermediate_size=512,
            v_head_dim=32,
            nope_head_dim=32,
            rope_head_dim=64,
            q_lora_rank=192,
            kv_lora_rank=64,
            max_seq_len=128,
        )
    
    def test_forward_shape(self, config):
        """Test block forward pass shape."""
        rngs = nnx.Rngs(0)
        block = MLABlock(**config, rngs=rngs)
        
        batch_size, seq_len = 2, 16
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, config["hidden_size"]))
        
        output, state = block(x)
        
        assert output.shape == x.shape
        assert state is None
    
    def test_residual_connection(self, config):
        """Test that residual connections work."""
        rngs = nnx.Rngs(0)
        block = MLABlock(**config, rngs=rngs)
        
        batch_size, seq_len = 2, 16
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, config["hidden_size"]))
        
        output, _ = block(x)
        
        # Output should not be identical to input (something happened)
        assert not jnp.allclose(output, x)
        
        # But should be similar magnitude (residual helps)
        assert jnp.abs(output).mean() < jnp.abs(x).mean() * 10


class TestApplyRopeMLA:
    """Tests for RoPE application in MLA context."""
    
    def test_rope_application(self):
        """Test that RoPE is applied correctly."""
        batch_size, n_heads, seq_len, head_dim = 2, 4, 8, 64
        
        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_heads, seq_len, head_dim))
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 1, seq_len, head_dim))
        
        # Create cos/sin
        positions = jnp.arange(seq_len)
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2) / head_dim))
        freqs = jnp.outer(positions, inv_freq)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        
        q_rot, k_rot = apply_rope_mla(q, k, cos, sin)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
        # Rotated vectors should have same norm (RoPE is orthogonal)
        np.testing.assert_allclose(
            jnp.linalg.norm(q, axis=-1),
            jnp.linalg.norm(q_rot, axis=-1),
            rtol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
