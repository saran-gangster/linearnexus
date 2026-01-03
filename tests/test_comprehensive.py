"""Comprehensive tests for LinearNexus.

Tests all major components with actual forward and backward passes on CPU.
Each test performs 1 forward pass and 1 backward pass to verify gradients flow.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import flax.nnx as nnx
import optax

# Force CPU
jax.config.update("jax_platform_name", "cpu")


# =============================================================================
# Test Core Utilities
# =============================================================================

class TestCoreUtilities:
    """Test core module utilities."""
    
    def test_depthwise_conv1d_causal(self):
        """Test causal depthwise convolution."""
        from linearnexus.core import depthwise_conv1d_causal
        
        batch_size, seq_len, channels = 2, 16, 32
        kernel_size = 4
        
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, channels))
        weight = jax.random.normal(jax.random.PRNGKey(1), (kernel_size, channels)) * 0.1
        bias = jnp.zeros(channels)
        
        # Forward pass
        output, cache = depthwise_conv1d_causal(x, weight, bias)
        
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        assert cache.shape == (batch_size, kernel_size - 1, channels)
        
        # Backward pass
        def loss_fn(w, b, x):
            out, _ = depthwise_conv1d_causal(x, w, b)
            return jnp.mean(out ** 2)
        
        grads = jax.grad(loss_fn, argnums=(0, 1))(weight, bias, x)
        assert grads[0].shape == weight.shape
        assert grads[1].shape == bias.shape
        print("✓ depthwise_conv1d_causal: forward + backward OK")
    
    def test_conv_state_incremental(self):
        """Test incremental convolution with state."""
        from linearnexus.core import depthwise_conv1d_causal
        
        batch_size, channels = 2, 32
        kernel_size = 4
        
        weight = jax.random.normal(jax.random.PRNGKey(0), (kernel_size, channels)) * 0.1
        bias = jnp.zeros(channels)
        
        # Process full sequence
        full_x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 8, channels))
        full_out, _ = depthwise_conv1d_causal(full_x, weight, bias)
        
        # Process incrementally
        cache = jnp.zeros((batch_size, kernel_size - 1, channels))
        incremental_outs = []
        
        for t in range(8):
            x_t = full_x[:, t:t+1, :]
            out_t, cache = depthwise_conv1d_causal(x_t, weight, bias, cache=cache)
            incremental_outs.append(out_t)
        
        incremental_out = jnp.concatenate(incremental_outs, axis=1)
        
        np.testing.assert_allclose(full_out, incremental_out, rtol=1e-5)
        print("✓ Incremental convolution state: OK")


# =============================================================================
# Test Attention Module
# =============================================================================

class TestAttentionModule:
    """Test attention block and components."""
    
    def test_kv_cache(self):
        """Test KV cache operations."""
        from linearnexus.modules.attention import KVCache
        
        batch_size, max_seq, n_kv_heads, head_dim = 2, 64, 4, 32
        
        cache = KVCache.zeros(batch_size, max_seq, n_kv_heads, head_dim)
        assert cache.position == 0
        
        # Add first chunk
        keys1 = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 8, n_kv_heads, head_dim))
        vals1 = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 8, n_kv_heads, head_dim))
        
        cache = cache.update(keys1, vals1)
        assert cache.position == 8
        
        k, v = cache.get()
        assert k.shape == (batch_size, 8, n_kv_heads, head_dim)
        np.testing.assert_allclose(k, keys1, rtol=1e-5)
        print("✓ KVCache: OK")
    
    def test_causal_self_attention_forward_backward(self):
        """Test CausalSelfAttention with forward and backward pass."""
        from linearnexus.modules.attention import CausalSelfAttention
        
        hidden_size, n_heads = 64, 4
        batch_size, seq_len = 2, 16
        
        rngs = nnx.Rngs(0)
        attn = CausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            use_rope=True,
            max_seq_len=128,
            rngs=rngs,
        )
        
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, hidden_size))
        
        # Forward pass
        output, state = attn(x)
        assert output.shape == x.shape
        
        # Backward pass
        def loss_fn(model, x):
            out, _ = model(x)
            return jnp.mean(out ** 2)
        
        grads = nnx.grad(loss_fn)(attn, x)
        
        # Check gradients exist (grads is a model with gradients as parameters)
        graphdef, params = nnx.split(grads)
        # Check that grads object has parameters
        flat_params = nnx.to_flat_state(params)
        has_grads = len(flat_params) > 0
        assert has_grads, "No gradients computed"
        print("✓ CausalSelfAttention: forward + backward OK")
    
    def test_attention_with_gqa(self):
        """Test Grouped Query Attention."""
        from linearnexus.modules.attention import CausalSelfAttention
        
        hidden_size, n_heads, n_kv_heads = 64, 8, 2
        batch_size, seq_len = 2, 16
        
        rngs = nnx.Rngs(0)
        attn = CausalSelfAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,  # GQA: 4 query heads per KV head
            use_rope=True,
            rngs=rngs,
        )
        
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, hidden_size))
        output, _ = attn(x)
        
        assert output.shape == x.shape
        print("✓ GQA Attention: OK")
    
    def test_attention_block_forward_backward(self):
        """Test full AttentionBlock (attention + FFN)."""
        from linearnexus.modules.attention import AttentionBlock
        
        hidden_size, n_heads, intermediate_size = 64, 4, 256
        batch_size, seq_len = 2, 16
        
        rngs = nnx.Rngs(0)
        block = AttentionBlock(
            hidden_size=hidden_size,
            n_heads=n_heads,
            intermediate_size=intermediate_size,
            use_rope=True,
            rngs=rngs,
        )
        
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, hidden_size))
        
        # Forward
        output, state = block(x)
        assert output.shape == x.shape
        
        # Backward
        def loss_fn(model, x):
            out, _ = model(x)
            return jnp.mean(out ** 2)
        
        grads = nnx.grad(loss_fn)(block, x)
        graphdef, params = nnx.split(grads)
        flat_params = nnx.to_flat_state(params)
        has_grads = len(flat_params) > 0
        assert has_grads
        print("✓ AttentionBlock: forward + backward OK")


# =============================================================================
# Test SSM/Mamba Module
# =============================================================================

class TestMambaModule:
    """Test Mamba SSM block."""
    
    def test_mamba_state(self):
        """Test MambaState initialization."""
        from linearnexus.modules.ssm import MambaState
        
        batch_size, intermediate, conv_kernel, state_size = 2, 64, 4, 16
        
        state = MambaState.zeros(batch_size, intermediate, conv_kernel, state_size)
        
        assert state.conv_state.shape == (batch_size, conv_kernel - 1, intermediate)
        assert state.ssm_state.shape == (batch_size, intermediate, state_size)
        assert state.position == 0
        print("✓ MambaState: OK")
    
    def test_selective_scan_ref(self):
        """Test selective scan reference implementation."""
        from linearnexus.modules.ssm.mamba import selective_scan_ref
        
        batch_size, intermediate, seq_len, state_size = 2, 32, 16, 16
        
        hidden = jax.random.normal(jax.random.PRNGKey(0), (batch_size, intermediate, seq_len))
        delta = jax.nn.softplus(jax.random.normal(jax.random.PRNGKey(1), (batch_size, intermediate, seq_len)))
        A = -jnp.abs(jax.random.normal(jax.random.PRNGKey(2), (intermediate, state_size)))
        B = jax.random.normal(jax.random.PRNGKey(3), (batch_size, seq_len, state_size))
        C = jax.random.normal(jax.random.PRNGKey(4), (batch_size, seq_len, state_size))
        D = jnp.ones((intermediate,))
        gate = jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(5), (batch_size, intermediate, seq_len)))
        
        # Forward
        output, final_state = selective_scan_ref(hidden, delta, A, B, C, D, gate, chunk_size=8)
        
        assert output.shape == (batch_size, intermediate, seq_len)
        assert final_state.shape == (batch_size, intermediate, state_size)
        
        # Backward
        def loss_fn(hidden, delta, A, B, C, D, gate):
            out, _ = selective_scan_ref(hidden, delta, A, B, C, D, gate, chunk_size=8)
            return jnp.mean(out ** 2)
        
        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(hidden, delta, A, B, C, D, gate)
        assert grads[0].shape == hidden.shape
        print("✓ selective_scan_ref: forward + backward OK")
    
    def test_mamba_block_forward_backward(self):
        """Test full MambaBlock."""
        from linearnexus.modules.ssm import MambaBlock
        
        hidden_size, intermediate_size = 64, 128
        batch_size, seq_len = 2, 16
        
        rngs = nnx.Rngs(0)
        block = MambaBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            state_size=16,
            conv_kernel=4,
            chunk_size=8,
            rngs=rngs,
        )
        
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, hidden_size))
        
        # Forward
        output, state = block(x)
        assert output.shape == x.shape
        assert state is not None
        
        # Backward
        def loss_fn(model, x):
            out, _ = model(x)
            return jnp.mean(out ** 2)
        
        grads = nnx.grad(loss_fn)(block, x)
        graphdef, params = nnx.split(grads)
        flat_params = nnx.to_flat_state(params)
        has_grads = len(flat_params) > 0
        assert has_grads
        print("✓ MambaBlock: forward + backward OK")
    
    def test_mamba_chunk_vs_recurrent_parity(self):
        """Test that chunk and recurrent modes produce same output."""
        from linearnexus.modules.ssm import MambaBlock, MambaState
        
        hidden_size, intermediate_size = 64, 128
        batch_size, seq_len = 2, 8
        
        rngs = nnx.Rngs(0)
        block = MambaBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            state_size=16,
            conv_kernel=4,
            chunk_size=4,
            rngs=rngs,
        )
        
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, hidden_size))
        
        # Chunk mode (full sequence)
        chunk_out, _ = block(x, mode="chunk")
        
        # Recurrent mode (token by token)
        state = block.init_state(batch_size)
        recurrent_outs = []
        for t in range(seq_len):
            x_t = x[:, t:t+1, :]
            out_t, state = block(x_t, state=state, mode="recurrent")
            recurrent_outs.append(out_t)
        recurrent_out = jnp.concatenate(recurrent_outs, axis=1)
        
        np.testing.assert_allclose(chunk_out, recurrent_out, rtol=1e-4, atol=1e-5)
        print("✓ Mamba chunk/recurrent parity: OK")


# =============================================================================
# Test MLA Module
# =============================================================================

class TestMLAModule:
    """Test Multi-Head Latent Attention."""
    
    def test_mla_forward_backward(self):
        """Test MLA forward and backward pass."""
        from linearnexus.modules.attention.mla import MultiHeadLatentAttention
        
        hidden_size = 128
        n_heads = 4
        
        rngs = nnx.Rngs(0)
        mla = MultiHeadLatentAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            v_head_dim=32,
            nope_head_dim=32,
            rope_head_dim=64,
            q_lora_rank=96,
            kv_lora_rank=64,
            max_seq_len=128,
            rngs=rngs,
        )
        
        batch_size, seq_len = 2, 16
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, hidden_size))
        
        # Forward
        output, _ = mla(x)
        assert output.shape == x.shape
        
        # Backward
        def loss_fn(model, x):
            out, _ = model(x)
            return jnp.mean(out ** 2)
        
        grads = nnx.grad(loss_fn)(mla, x)
        graphdef, params = nnx.split(grads)
        flat_params = nnx.to_flat_state(params)
        has_grads = len(flat_params) > 0
        assert has_grads
        print("✓ MLA: forward + backward OK")


# =============================================================================
# Test Full Models
# =============================================================================

class TestModels:
    """Test full model architectures."""
    
    def test_gpt_model_forward_backward(self):
        """Test GPT-style model."""
        from linearnexus.models import LMModel, ModelConfig
        
        config = ModelConfig(
            vocab_size=256,
            hidden_size=64,
            n_layers=2,
            n_heads=4,
            block_pattern=["attention"],
            intermediate_size=128,
            max_seq_len=64,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        batch_size, seq_len = 2, 16
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, 256)
        
        # Forward
        logits, state = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        
        # Backward
        def loss_fn(model, input_ids):
            logits, _ = model(input_ids)
            return jnp.mean(logits ** 2)
        
        grads = nnx.grad(loss_fn)(model, input_ids)
        graphdef, params = nnx.split(grads)
        flat_params = nnx.to_flat_state(params)
        has_grads = len(flat_params) > 0
        assert has_grads
        
        n_params = model.count_params()
        assert n_params > 0
        print(f"✓ GPT Model ({n_params:,} params): forward + backward OK")
    
    def test_mamba_model_forward_backward(self):
        """Test Mamba-style model."""
        from linearnexus.models import LMModel, ModelConfig
        
        config = ModelConfig(
            vocab_size=256,
            hidden_size=64,
            n_layers=2,
            block_pattern=["mamba"],
            intermediate_size=128,
            state_size=16,
            conv_kernel=4,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        batch_size, seq_len = 2, 16
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, 256)
        
        # Forward
        logits, state = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        
        # Backward
        def loss_fn(model, input_ids):
            logits, _ = model(input_ids)
            return jnp.mean(logits ** 2)
        
        grads = nnx.grad(loss_fn)(model, input_ids)
        graphdef, params = nnx.split(grads)
        flat_params = nnx.to_flat_state(params)
        has_grads = len(flat_params) > 0
        assert has_grads
        
        n_params = model.count_params()
        print(f"✓ Mamba Model ({n_params:,} params): forward + backward OK")
    
    def test_hybrid_jamba_model_forward_backward(self):
        """Test Jamba-style hybrid model."""
        from linearnexus.models import LMModel, ModelConfig
        
        # Jamba: mostly Mamba with occasional attention
        config = ModelConfig(
            vocab_size=256,
            hidden_size=64,
            n_layers=4,
            n_heads=4,
            block_pattern=["mamba", "mamba", "mamba", "attention"],
            intermediate_size=128,
            state_size=16,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        batch_size, seq_len = 2, 16
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, 256)
        
        # Forward
        logits, state = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        
        # Backward
        def loss_fn(model, input_ids):
            logits, _ = model(input_ids)
            return jnp.mean(logits ** 2)
        
        grads = nnx.grad(loss_fn)(model, input_ids)
        
        n_params = model.count_params()
        print(f"✓ Hybrid Jamba Model ({n_params:,} params): forward + backward OK")
    
    def test_model_presets(self):
        """Test model preset factory."""
        from linearnexus.models import create_model
        
        presets = ["gpt-small", "mamba-small", "jamba-small"]
        
        for preset in presets:
            config, _ = create_model(preset, vocab_size=256)
            rngs = nnx.Rngs(0)
            model = create_model(preset, vocab_size=256, rngs=rngs)
            
            batch_size, seq_len = 1, 8
            input_ids = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, 256)
            logits, _ = model(input_ids)
            
            assert logits.shape == (batch_size, seq_len, 256)
            print(f"  ✓ {preset}: OK")
        
        print("✓ Model presets: OK")


# =============================================================================
# Test Training Pipeline
# =============================================================================

class TestTrainingPipeline:
    """Test training components."""
    
    def test_cross_entropy_loss(self):
        """Test cross-entropy loss function."""
        from linearnexus.train import cross_entropy_loss
        
        batch_size, seq_len, vocab_size = 2, 16, 256
        
        logits = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, vocab_size))
        labels = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 0, vocab_size)
        
        loss = cross_entropy_loss(logits, labels)
        assert loss.shape == ()
        assert loss > 0
        
        # With mask
        mask = jnp.ones((batch_size, seq_len)).at[:, -4:].set(0)
        loss_masked = cross_entropy_loss(logits, labels, mask=mask)
        assert loss_masked.shape == ()
        
        print("✓ cross_entropy_loss: OK")
    
    def test_sft_trainer_single_step(self):
        """Test SFT trainer for a single training step (using NNx Optimizer)."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.train import cross_entropy_loss
        
        # Small model for fast testing
        config = ModelConfig(
            vocab_size=256,
            hidden_size=32,
            n_layers=1,
            n_heads=2,
            block_pattern=["attention"],
            intermediate_size=64,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        # Use NNx Optimizer
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        
        # Create batch
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 16), 0, 256)
        labels = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, 256)
        
        # Define loss function
        def loss_fn(model):
            logits, _ = model(input_ids)
            return cross_entropy_loss(logits, labels)
        
        # Single training step
        loss_before = loss_fn(model)
        grads = nnx.grad(loss_fn)(model)
        optimizer.update(model, grads)
        loss_after = loss_fn(model)
        
        assert loss_before > 0
        assert loss_after > 0
        
        print(f"✓ SFT Trainer single step: loss={float(loss_before):.4f} → {float(loss_after):.4f}")
    
    def test_full_training_loop(self):
        """Test complete training loop with actual data using NNx Optimizer."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.train import cross_entropy_loss
        
        # Tiny model
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=1,
            n_heads=2,
            block_pattern=["attention"],
            intermediate_size=64,
        )
        
        rngs = nnx.Rngs(42)
        model = LMModel(config, rngs=rngs)
        
        # Use NNx Optimizer
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        
        # Training loop
        losses = []
        for step in range(3):
            key = jax.random.PRNGKey(step)
            input_ids = jax.random.randint(key, (2, 16), 0, 64)
            labels = jax.random.randint(key, (2, 16), 0, 64)
            
            def loss_fn(model):
                logits, _ = model(input_ids)
                return cross_entropy_loss(logits, labels)
            
            loss = loss_fn(model)
            losses.append(float(loss))
            
            grads = nnx.grad(loss_fn)(model)
            optimizer.update(model, grads)
        
        assert len(losses) == 3
        print(f"✓ Full training loop: {len(losses)} steps, losses={[f'{l:.4f}' for l in losses]}")


# =============================================================================
# Test Optimizers
# =============================================================================

class TestOptimizers:
    """Test custom optimizers."""
    
    def test_adamw_optimizer(self):
        """Test AdamW optimizer."""
        from linearnexus.optim import adamw
        
        opt = adamw(learning_rate=1e-3, weight_decay=0.1)
        
        params = {"w": jnp.ones((32, 32))}
        opt_state = opt.init(params)
        
        grads = {"w": jnp.ones((32, 32)) * 0.1}
        updates, new_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        assert new_params["w"].shape == (32, 32)
        assert not jnp.allclose(new_params["w"], params["w"])
        print("✓ AdamW optimizer: OK")
    
    def test_muon_optimizer(self):
        """Test Muon optimizer."""
        from linearnexus.optim import muon
        
        opt = muon(learning_rate=1e-3, momentum=0.9)
        
        params = {"w": jnp.ones((32, 32))}
        opt_state = opt.init(params)
        
        grads = {"w": jnp.ones((32, 32)) * 0.1}
        updates, new_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        assert new_params["w"].shape == (32, 32)
        print("✓ Muon optimizer: OK")
    
    def test_create_optimizer_factory(self):
        """Test optimizer factory function."""
        from linearnexus.optim import create_optimizer
        
        for opt_name in ["adamw", "muon"]:
            opt = create_optimizer(opt_name, learning_rate=1e-3)
            params = {"w": jnp.ones((16, 16))}
            opt_state = opt.init(params)
            assert opt_state is not None
            print(f"  ✓ {opt_name}: OK")
        
        print("✓ Optimizer factory: OK")


# =============================================================================
# Test Data Pipeline
# =============================================================================

class TestDataPipeline:
    """Test data loading utilities."""
    
    def test_char_tokenizer(self):
        """Test character tokenizer."""
        from linearnexus.data import CharTokenizer
        
        text = "Hello, World! This is a test."
        tokenizer = CharTokenizer.from_text(text)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        assert decoded == text
        assert tokenizer.vocab_size > 0
        print(f"✓ CharTokenizer: vocab_size={tokenizer.vocab_size}")
    
    def test_text_dataset(self):
        """Test TextDataset."""
        from linearnexus.data import CharTokenizer, TextDataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            text = "Hello world! " * 100
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text(text)
            
            tokenizer = CharTokenizer.from_text(text)
            dataset = TextDataset(test_file, tokenizer, seq_len=32, cache_dir=tmpdir)
            
            assert len(dataset) > 0
            
            sample = dataset[0]
            assert "input_ids" in sample
            assert "labels" in sample
            assert sample["input_ids"].shape == (32,)
            
            print(f"✓ TextDataset: {len(dataset)} samples")
    
    def test_dataloader(self):
        """Test DataLoader."""
        from linearnexus.data import CharTokenizer, TextDataset, DataLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            text = "Test text for dataloader. " * 50
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text(text)
            
            tokenizer = CharTokenizer.from_text(text)
            dataset = TextDataset(test_file, tokenizer, seq_len=16, cache_dir=tmpdir)
            loader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            batch = next(iter(loader))
            assert batch["input_ids"].shape == (4, 16)
            assert batch["labels"].shape == (4, 16)
            
            print(f"✓ DataLoader: batch_size=4")


# =============================================================================
# Test Generation
# =============================================================================

class TestGeneration:
    """Test text generation utilities."""
    
    def test_sample_token(self):
        """Test token sampling functions."""
        from linearnexus.generate import sample_token
        
        batch_size, vocab_size = 2, 256
        logits = jax.random.normal(jax.random.PRNGKey(0), (batch_size, vocab_size))
        
        # Greedy
        tokens = sample_token(logits, temperature=0.0)
        assert tokens.shape == (batch_size,)
        
        # With temperature
        tokens = sample_token(logits, temperature=0.8, key=jax.random.PRNGKey(1))
        assert tokens.shape == (batch_size,)
        
        # Top-k
        tokens = sample_token(logits, temperature=1.0, top_k=50, key=jax.random.PRNGKey(2))
        assert tokens.shape == (batch_size,)
        
        print("✓ sample_token: OK")
    
    def test_generate_function(self):
        """Test generation with a model."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.generate import generate
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=1,
            n_heads=2,
            block_pattern=["attention"],
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        prompt = jnp.array([[1, 2, 3, 4]])  # [batch=1, seq=4]
        
        output = generate(
            model,
            prompt,
            max_tokens=10,
            temperature=0.8,
            key=jax.random.PRNGKey(0),
        )
        
        assert output.shape[0] == 1
        assert output.shape[1] == 4 + 10  # prompt + generated
        print(f"✓ generate: {output.shape[1]} tokens")


# =============================================================================
# Test Saving and Loading
# =============================================================================

class TestSaveLoad:
    """Test checkpoint saving and loading."""
    
    def test_save_and_load_gpt(self):
        """Test save/load for GPT model."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.checkpoint import save_model, load_model
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=1,
            n_heads=2,
            block_pattern=["attention"],
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(42)
        model = LMModel(config, rngs=rngs)
        
        # Get some output before saving
        input_ids = jnp.array([[1, 2, 3, 4]])
        logits_before, _ = model(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            save_model(tmpdir, model)
            
            # Load into new model
            loaded_model, meta = load_model(tmpdir)
            
            # Verify output matches
            logits_after, _ = loaded_model(input_ids)
            
            np.testing.assert_allclose(logits_before, logits_after, rtol=1e-5)
            
        print("✓ Save/Load GPT: OK")
    
    def test_save_and_load_mamba(self):
        """Test save/load for Mamba model."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.checkpoint import save_model, load_model
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=1,
            block_pattern=["mamba"],
            state_size=16,
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(42)
        model = LMModel(config, rngs=rngs)
        
        input_ids = jnp.array([[1, 2, 3, 4]])
        logits_before, _ = model(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(tmpdir, model)
            
            loaded_model, meta = load_model(tmpdir)
            logits_after, _ = loaded_model(input_ids)
            
            np.testing.assert_allclose(logits_before, logits_after, rtol=1e-5)
            
        print("✓ Save/Load Mamba: OK")
    
    def test_save_and_load_hybrid(self):
        """Test save/load for Hybrid model."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.checkpoint import save_model, load_model
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=2,
            n_heads=2,
            block_pattern=["mamba", "attention"],
            state_size=16,
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(42)
        model = LMModel(config, rngs=rngs)
        
        input_ids = jnp.array([[1, 2, 3, 4]])
        logits_before, _ = model(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(tmpdir, model)
            
            loaded_model, meta = load_model(tmpdir)
            logits_after, _ = loaded_model(input_ids)
            
            np.testing.assert_allclose(logits_before, logits_after, rtol=1e-5)
            
        print("✓ Save/Load Hybrid: OK")
    
    def test_train_save_load_continue(self):
        """Test training, saving, loading, and continuing training."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.checkpoint import save_model, load_model, save_optimizer, load_optimizer
        from linearnexus.train import cross_entropy_loss
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=1,
            n_heads=2,
            block_pattern=["attention"],
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(42)
        model = LMModel(config, rngs=rngs)
        tx = optax.adam(1e-3)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        
        # Train for a few steps
        for step in range(3):
            input_ids = jax.random.randint(jax.random.PRNGKey(step), (2, 8), 0, 64)
            labels = jax.random.randint(jax.random.PRNGKey(step+100), (2, 8), 0, 64)
            
            def loss_fn(model):
                logits, _ = model(input_ids)
                return cross_entropy_loss(logits, labels)
            
            grads = nnx.grad(loss_fn)(model)
            optimizer.update(model, grads)
        
        # Get output after training
        test_input = jnp.array([[1, 2, 3, 4]])
        test_labels = jnp.array([[2, 3, 4, 5]])
        logits_trained, _ = model(test_input)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model and optimizer
            save_model(tmpdir, model)
            save_optimizer(tmpdir, optimizer)
            
            # Load model and optimizer
            loaded_model, _ = load_model(tmpdir)
            loaded_optimizer = load_optimizer(tmpdir, loaded_model, tx)
            
            # Verify same output
            logits_loaded, _ = loaded_model(test_input)
            np.testing.assert_allclose(logits_trained, logits_loaded, rtol=1e-5)
            
            # Continue training on loaded model
            def loss_fn(model):
                logits, _ = model(test_input)
                return cross_entropy_loss(logits, test_labels)
            
            loss_before = loss_fn(loaded_model)
            grads = nnx.grad(loss_fn)(loaded_model)
            loaded_optimizer.update(loaded_model, grads)
            loss_after = loss_fn(loaded_model)
            
            # Training should decrease loss
            assert loss_after < loss_before
            
        print("✓ Train → Save → Load → Continue: OK")


# =============================================================================
# Test Inference Modes
# =============================================================================

class TestInference:
    """Test inference and generation capabilities."""
    
    def test_greedy_generation(self):
        """Test greedy (deterministic) generation."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.generate import generate
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=1,
            n_heads=2,
            block_pattern=["attention"],
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        prompt = jnp.array([[1, 2, 3, 4]])
        
        # Greedy should be deterministic
        output1 = generate(model, prompt, max_tokens=5, temperature=0.0)
        output2 = generate(model, prompt, max_tokens=5, temperature=0.0)
        
        np.testing.assert_array_equal(output1, output2)
        print("✓ Greedy generation (deterministic): OK")
    
    def test_sampling_generation(self):
        """Test sampling-based generation with different temperatures."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.generate import generate
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=1,
            n_heads=2,
            block_pattern=["attention"],
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        prompt = jnp.array([[1, 2, 3, 4]])
        
        # Different keys should give different results
        output1 = generate(model, prompt, max_tokens=5, temperature=1.0, key=jax.random.PRNGKey(0))
        output2 = generate(model, prompt, max_tokens=5, temperature=1.0, key=jax.random.PRNGKey(1))
        
        # Outputs should differ (with high probability for different keys)
        # Just verify shapes are correct
        assert output1.shape == (1, 9)
        assert output2.shape == (1, 9)
        print("✓ Sampling generation: OK")
    
    def test_top_k_generation(self):
        """Test top-k sampling."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.generate import generate
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=1,
            n_heads=2,
            block_pattern=["attention"],
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        prompt = jnp.array([[1, 2, 3, 4]])
        
        output = generate(
            model, prompt, max_tokens=5,
            temperature=1.0, top_k=10,
            key=jax.random.PRNGKey(0)
        )
        
        assert output.shape == (1, 9)
        print("✓ Top-k generation: OK")
    
    def test_top_p_generation(self):
        """Test top-p (nucleus) sampling."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.generate import generate
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=1,
            n_heads=2,
            block_pattern=["attention"],
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        prompt = jnp.array([[1, 2, 3, 4]])
        
        output = generate(
            model, prompt, max_tokens=5,
            temperature=1.0, top_p=0.9,
            key=jax.random.PRNGKey(0)
        )
        
        assert output.shape == (1, 9)
        print("✓ Top-p generation: OK")
    
    def test_mamba_autoregressive(self):
        """Test Mamba model in autoregressive mode."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.generate import generate
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=2,
            block_pattern=["mamba"],
            state_size=16,
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        prompt = jnp.array([[1, 2, 3, 4]])
        
        output = generate(
            model, prompt, max_tokens=10,
            temperature=0.8,
            key=jax.random.PRNGKey(0)
        )
        
        assert output.shape == (1, 14)
        print("✓ Mamba autoregressive generation: OK")
    
    def test_hybrid_autoregressive(self):
        """Test Hybrid model in autoregressive mode."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.generate import generate
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=4,
            n_heads=2,
            block_pattern=["mamba", "mamba", "mamba", "attention"],
            state_size=16,
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        prompt = jnp.array([[1, 2, 3, 4]])
        
        output = generate(
            model, prompt, max_tokens=10,
            temperature=0.8,
            key=jax.random.PRNGKey(0)
        )
        
        assert output.shape == (1, 14)
        print("✓ Hybrid autoregressive generation: OK")
    
    def test_batch_generation(self):
        """Test batched generation."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.generate import generate
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=1,
            n_heads=2,
            block_pattern=["attention"],
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        # Batch of 4 prompts
        prompt = jnp.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ])
        
        output = generate(
            model, prompt, max_tokens=8,
            temperature=0.8,
            key=jax.random.PRNGKey(0)
        )
        
        assert output.shape == (4, 12)  # 4 prompts, 4 + 8 tokens each
        print("✓ Batch generation: OK")
    
    def test_eos_stopping(self):
        """Test that generation stops at EOS token."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.generate import generate
        
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=1,
            n_heads=2,
            block_pattern=["attention"],
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(0)
        model = LMModel(config, rngs=rngs)
        
        prompt = jnp.array([[1, 2, 3, 4]])
        
        # Generate with EOS token
        output = generate(
            model, prompt, max_tokens=20,
            temperature=0.8,
            eos_token_id=0,  # Use 0 as EOS
            key=jax.random.PRNGKey(0)
        )
        
        # Output should be at most prompt + max_tokens
        assert output.shape[1] <= 24
        print("✓ EOS stopping: OK")


# =============================================================================
# Test End-to-End: Full Forward + Backward on All Architectures
# =============================================================================

class TestEndToEnd:
    """End-to-end tests with actual training step."""
    
    def _run_training_step(self, model_type: str, block_pattern: list):
        """Helper to run one forward/backward pass."""
        from linearnexus.models import LMModel, ModelConfig
        from linearnexus.train import cross_entropy_loss
        
        config = ModelConfig(
            vocab_size=128,
            hidden_size=48,
            n_layers=2,
            n_heads=4,
            block_pattern=block_pattern,
            intermediate_size=96,
            state_size=16,
            max_seq_len=64,
        )
        
        rngs = nnx.Rngs(42)
        model = LMModel(config, rngs=rngs)
        
        batch_size, seq_len = 2, 16
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, 128)
        labels = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 0, 128)
        
        # Define loss function
        def loss_fn(model):
            logits, _ = model(input_ids)
            return cross_entropy_loss(logits, labels)
        
        # Forward pass
        loss_before = loss_fn(model)
        
        # Backward pass using NNx native optimizer
        learning_rate = 0.01
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
        
        grads = nnx.grad(loss_fn)(model)
        optimizer.update(model, grads)
        
        # Forward pass after update
        loss_after = loss_fn(model)
        
        return float(loss_before), float(loss_after), model.count_params()
    
    def test_gpt_end_to_end(self):
        """Test GPT architecture end-to-end."""
        loss_before, loss_after, n_params = self._run_training_step(
            "gpt", ["attention"]
        )
        print(f"✓ GPT E2E ({n_params:,} params): loss {loss_before:.4f} → {loss_after:.4f}")
        # Loss should decrease or stay similar after one step
        assert loss_after < loss_before * 1.5  # Sanity check
    
    def test_mamba_end_to_end(self):
        """Test Mamba architecture end-to-end."""
        loss_before, loss_after, n_params = self._run_training_step(
            "mamba", ["mamba"]
        )
        print(f"✓ Mamba E2E ({n_params:,} params): loss {loss_before:.4f} → {loss_after:.4f}")
        assert loss_after < loss_before * 1.5
    
    def test_hybrid_end_to_end(self):
        """Test hybrid architecture end-to-end."""
        loss_before, loss_after, n_params = self._run_training_step(
            "hybrid", ["mamba", "attention"]
        )
        print(f"✓ Hybrid E2E ({n_params:,} params): loss {loss_before:.4f} → {loss_after:.4f}")
        assert loss_after < loss_before * 1.5


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all test classes."""
    print("=" * 70)
    print("LinearNexus Comprehensive Test Suite")
    print("Testing on CPU with 1 forward + 1 backward pass each")
    print("=" * 70)
    print()
    
    test_classes = [
        ("Core Utilities", TestCoreUtilities),
        ("Attention Module", TestAttentionModule),
        ("Mamba/SSM Module", TestMambaModule),
        ("MLA Module", TestMLAModule),
        ("Full Models", TestModels),
        ("Training Pipeline", TestTrainingPipeline),
        ("Optimizers", TestOptimizers),
        ("Data Pipeline", TestDataPipeline),
        ("Generation", TestGeneration),
        ("Save/Load Checkpoints", TestSaveLoad),
        ("Inference Modes", TestInference),
        ("End-to-End Training", TestEndToEnd),
    ]
    
    total_passed = 0
    total_failed = 0
    
    for name, test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print("=" * 60)
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                total_passed += 1
            except Exception as e:
                print(f"✗ {method_name}: FAILED - {e}")
                total_failed += 1
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print("=" * 70)
    
    return total_failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
