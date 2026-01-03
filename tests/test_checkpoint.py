"""Tests for the checkpoint module."""

import tempfile
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import optax
import pytest

from linearnexus.checkpoint import (
    CheckpointManager,
    CheckpointState,
    save_model,
    load_model,
    save_optimizer,
    load_optimizer,
    save_checkpoint,
    load_checkpoint,
)
from linearnexus.models import LMModel, ModelConfig


class TestSaveLoadModel:
    """Test basic model save/load functionality."""
    
    def test_save_load_gpt(self):
        """Test save/load for GPT model."""
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=2,
            n_heads=2,
            block_pattern=["attention"],
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(42)
        model = LMModel(config, rngs=rngs)
        
        # Get output before saving
        input_ids = jnp.array([[1, 2, 3, 4]])
        logits_before, _ = model(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_model(tmpdir, model, {"model_config": config.to_dict()})
            
            # Load into new model
            loaded_model, _ = load_model(tmpdir)
            logits_after, _ = loaded_model(input_ids)
            
            np.testing.assert_allclose(logits_before, logits_after, rtol=1e-5)
        
        print("✓ Save/Load GPT model: OK")
    
    def test_save_load_mamba(self):
        """Test save/load for Mamba model."""
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=2,
            block_pattern=["mamba"],
            state_size=16,
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(42)
        model = LMModel(config, rngs=rngs)
        
        input_ids = jnp.array([[1, 2, 3, 4]])
        logits_before, _ = model(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(tmpdir, model, {"model_config": config.to_dict()})
            loaded_model, _ = load_model(tmpdir)
            logits_after, _ = loaded_model(input_ids)
            
            np.testing.assert_allclose(logits_before, logits_after, rtol=1e-5)
        
        print("✓ Save/Load Mamba model: OK")
    
    def test_save_load_hybrid(self):
        """Test save/load for Hybrid model."""
        config = ModelConfig(
            vocab_size=64,
            hidden_size=32,
            n_layers=4,
            n_heads=2,
            block_pattern=["mamba", "mamba", "mamba", "attention"],
            state_size=16,
            max_seq_len=32,
        )
        
        rngs = nnx.Rngs(42)
        model = LMModel(config, rngs=rngs)
        
        input_ids = jnp.array([[1, 2, 3, 4]])
        logits_before, _ = model(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(tmpdir, model, {"model_config": config.to_dict()})
            loaded_model, _ = load_model(tmpdir)
            logits_after, _ = loaded_model(input_ids)
            
            np.testing.assert_allclose(logits_before, logits_after, rtol=1e-5)
        
        print("✓ Save/Load Hybrid model: OK")
    
    def test_save_load_compressed(self):
        """Test compressed vs uncompressed saving."""
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
        input_ids = jnp.array([[1, 2, 3, 4]])
        logits_before, _ = model(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save compressed (model.config auto-extracted)
            save_model(f"{tmpdir}/compressed", model, compress=True)
            # Save uncompressed
            save_model(f"{tmpdir}/uncompressed", model, compress=False)
            
            # Both should load correctly
            model1, _ = load_model(f"{tmpdir}/compressed")
            model2, _ = load_model(f"{tmpdir}/uncompressed")
            
            logits1, _ = model1(input_ids)
            logits2, _ = model2(input_ids)
            
            np.testing.assert_allclose(logits_before, logits1, rtol=1e-5)
            np.testing.assert_allclose(logits_before, logits2, rtol=1e-5)
        
        print("✓ Compressed/uncompressed saving: OK")


class TestSaveLoadOptimizer:
    """Test optimizer save/load functionality."""
    
    def test_save_load_adamw_optimizer(self):
        """Test save/load for AdamW optimizer."""
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
        tx = optax.adamw(1e-3)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        
        # Do some updates to populate optimizer state
        input_ids = jnp.array([[1, 2, 3, 4]])
        labels = jnp.array([[2, 3, 4, 5]])
        
        def loss_fn(model):
            logits, _ = model(input_ids)
            return jnp.mean((logits - jax.nn.one_hot(labels, 64)) ** 2)
        
        for _ in range(3):
            grads = nnx.grad(loss_fn)(model)
            optimizer.update(model, grads)
        
        # Get output after training
        logits_trained, _ = model(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model and optimizer
            save_model(tmpdir, model, {"model_config": config.to_dict()})
            save_optimizer(tmpdir, optimizer)
            
            # Load into new model/optimizer
            loaded_model, _ = load_model(tmpdir)
            loaded_optimizer = load_optimizer(tmpdir, loaded_model, tx)
            
            # Verify model output matches
            logits_loaded, _ = loaded_model(input_ids)
            np.testing.assert_allclose(logits_trained, logits_loaded, rtol=1e-5)
            
            # Continue training with loaded optimizer
            grads = nnx.grad(loss_fn)(loaded_model)
            loaded_optimizer.update(model, grads)
        
        print("✓ Save/Load optimizer: OK")


class TestCheckpointManager:
    """Test CheckpointManager functionality."""
    
    def test_basic_save_load(self):
        """Test basic save/load with manager."""
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
        
        input_ids = jnp.array([[1, 2, 3, 4]])
        logits_before, _ = model(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_to_keep=5)
            
            # Save
            manager.save(
                step=100,
                model=model,
                config={"model_config": config.to_dict()},
                metrics={"loss": 1.5},
            )
            
            assert len(manager) == 1
            assert manager.latest_step == 100
            
            # Load
            state = manager.load_latest()
            assert state is not None
            assert state.step == 100
            assert state.metrics["loss"] == 1.5
            
            logits_after, _ = state.model(input_ids)
            np.testing.assert_allclose(logits_before, logits_after, rtol=1e-5)
        
        print("✓ CheckpointManager basic save/load: OK")
    
    def test_max_to_keep(self):
        """Test automatic cleanup of old checkpoints."""
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_to_keep=3)
            
            # Save 5 checkpoints (model.config auto-extracted)
            for step in [100, 200, 300, 400, 500]:
                manager.save(step=step, model=model)
            
            # Should only keep 3
            assert len(manager) == 3
            
            # Should keep the latest 3
            steps = [c.step for c in manager.checkpoints]
            assert steps == [300, 400, 500]
        
        print("✓ CheckpointManager max_to_keep: OK")
    
    def test_best_checkpoint_tracking(self):
        """Test tracking best checkpoints by metric."""
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                tmpdir,
                max_to_keep=2,
                best_metric="loss",
                best_mode="min",
                best_to_keep=2,
            )
            
            # Save with varying loss
            losses = [2.0, 1.5, 1.8, 1.2, 1.6]
            for i, loss in enumerate(losses):
                manager.save(
                    step=(i + 1) * 100,
                    model=model,
                    metrics={"loss": loss},
                )
            
            # Best should be step 400 (loss=1.2)
            assert manager.best_step == 400
            
            # Load best
            state = manager.load_best()
            assert state is not None
            assert state.metrics["loss"] == 1.2
        
        print("✓ CheckpointManager best tracking: OK")
    
    def test_load_specific_step(self):
        """Test loading a specific step."""
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_to_keep=5)
            
            for step in [100, 200, 300]:
                manager.save(step=step, model=model, metrics={"step_val": step})
            
            # Load step 200
            state = manager.load_step(200)
            assert state is not None
            assert state.step == 200
            assert state.metrics["step_val"] == 200
        
        print("✓ CheckpointManager load specific step: OK")
    
    def test_discover_existing_checkpoints(self):
        """Test discovering checkpoints from existing directory."""
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoints with first manager (model.config auto-extracted)
            manager1 = CheckpointManager(tmpdir, max_to_keep=10)
            for step in [100, 200, 300]:
                manager1.save(step=step, model=model)
            
            # New manager should discover them
            manager2 = CheckpointManager(tmpdir, max_to_keep=10)
            assert len(manager2) == 3
            assert manager2.latest_step == 300
        
        print("✓ CheckpointManager discover existing: OK")
    
    def test_with_optimizer(self):
        """Test saving/loading with optimizer state."""
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
        tx = optax.adamw(1e-3)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        
        # Train a bit
        input_ids = jnp.array([[1, 2, 3, 4]])
        labels = jnp.array([[2, 3, 4, 5]])
        
        def loss_fn(model):
            logits, _ = model(input_ids)
            return jnp.mean((logits - jax.nn.one_hot(labels, 64)) ** 2)
        
        for _ in range(3):
            grads = nnx.grad(loss_fn)(model)
            optimizer.update(model, grads)
        
        logits_trained, _ = model(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            
            # Save with optimizer
            manager.save(
                step=100,
                model=model,
                optimizer=optimizer,
                config={"model_config": config.to_dict()},
            )
            
            # Load with optimizer
            state = manager.load_latest(tx=tx, should_load_optimizer=True)
            
            assert state.optimizer is not None
            
            logits_loaded, _ = state.model(input_ids)
            np.testing.assert_allclose(logits_trained, logits_loaded, rtol=1e-5)
            
            # Continue training
            grads = nnx.grad(loss_fn)(state.model)
            state.optimizer.update(state.model, grads)
        
        print("✓ CheckpointManager with optimizer: OK")


class TestConvenienceFunctions:
    """Test convenience save/load functions."""
    
    def test_save_load_checkpoint(self):
        """Test save_checkpoint and load_checkpoint convenience functions."""
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
        
        input_ids = jnp.array([[1, 2, 3, 4]])
        logits_before, _ = model(input_ids)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_checkpoint(
                path=tmpdir,
                model=model,
                step=100,
                config={"model_config": config.to_dict()},
                metrics={"loss": 1.5},
            )
            
            # Load
            loaded_model, meta = load_checkpoint(tmpdir)
            logits_after, _ = loaded_model(input_ids)
            
            np.testing.assert_allclose(logits_before, logits_after, rtol=1e-5)
        
        print("✓ Convenience save/load: OK")


def run_all_checkpoint_tests():
    """Run all checkpoint tests."""
    print("=" * 60)
    print("Checkpoint Module Tests")
    print("=" * 60)
    print()
    
    test_classes = [
        ("Save/Load Model", TestSaveLoadModel),
        ("Save/Load Optimizer", TestSaveLoadOptimizer),
        ("CheckpointManager", TestCheckpointManager),
        ("Convenience Functions", TestConvenienceFunctions),
    ]
    
    total_passed = 0
    total_failed = 0
    
    for name, test_class in test_classes:
        print(f"\n{'-'*40}")
        print(f"Testing: {name}")
        print("-" * 40)
        
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
    
    print("\n" + "=" * 60)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    return total_failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_checkpoint_tests()
    sys.exit(0 if success else 1)
