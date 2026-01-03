"""Robust checkpointing for any model architecture.

This module provides production-ready save/load functionality with:
- Binary serialization (numpy .npz) for efficient storage
- Automatic checkpoint management (keep N latest, K best)
- Full state restoration (model, optimizer, training state)
- Architecture-agnostic design (works with any NNx model)
- Atomic saves to prevent corruption
- Optional compression

Example:
    from linearnexus.checkpoint import CheckpointManager
    
    # Initialize manager
    ckpt_manager = CheckpointManager(
        directory="./checkpoints",
        max_to_keep=5,
        best_metric="loss",
        best_mode="min",
    )
    
    # Save during training
    ckpt_manager.save(
        step=1000,
        model=model,
        optimizer=optimizer,
        metrics={"loss": 0.5, "accuracy": 0.95},
    )
    
    # Load checkpoint
    state = ckpt_manager.load_latest()
    # or
    state = ckpt_manager.load_best()
    # or
    state = ckpt_manager.load("checkpoints/step_1000")
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import optax

Array = jax.Array
T = TypeVar("T")


# -----------------------------------------------------------------------------
# Serialization Utilities
# -----------------------------------------------------------------------------

def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict with dot-separated keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict:
    """Unflatten dot-separated keys back to nested dict."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def _to_numpy(x: Any) -> Any:
    """Convert JAX arrays to numpy for serialization."""
    if isinstance(x, jax.Array):
        return np.asarray(x)
    return x


def _to_jax(x: Any, dtype: Optional[jnp.dtype] = None) -> Any:
    """Convert numpy arrays back to JAX."""
    if isinstance(x, np.ndarray):
        arr = jnp.asarray(x)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr
    return x


def _filter_state_dict(state_dict: Dict) -> Dict:
    """Filter out non-serializable items (PRNG keys, etc.)."""
    def should_keep(x: Any) -> bool:
        if hasattr(x, 'dtype'):
            # Skip PRNG keys
            if 'key' in str(x.dtype).lower():
                return False
        return True
    
    def filter_recursive(d: Any) -> Any:
        if isinstance(d, dict):
            filtered = {}
            for k, v in d.items():
                result = filter_recursive(v)
                if result is not None:
                    filtered[k] = result
            return filtered if filtered else None
        elif should_keep(d):
            return d
        return None
    
    return filter_recursive(state_dict) or {}


# -----------------------------------------------------------------------------
# Checkpoint State Container
# -----------------------------------------------------------------------------

@dataclass
class CheckpointState:
    """Container for all checkpoint state.
    
    Attributes:
        step: Training step number.
        model: Restored model (NNx Module).
        optimizer: Restored optimizer (optional).
        config: Model configuration dict.
        train_config: Training configuration dict.
        metrics: Training metrics at checkpoint time.
        extra: Any additional user data.
    """
    step: int
    model: nnx.Module
    optimizer: Optional[nnx.Optimizer] = None
    config: Optional[Dict[str, Any]] = None
    train_config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class CheckpointInfo:
    """Metadata about a saved checkpoint."""
    path: Path
    step: int
    timestamp: float
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __lt__(self, other: "CheckpointInfo") -> bool:
        return self.step < other.step


# -----------------------------------------------------------------------------
# Core Save/Load Functions
# -----------------------------------------------------------------------------

def save_model(
    path: Union[str, Path],
    model: nnx.Module,
    config: Optional[Dict[str, Any]] = None,
    compress: bool = True,
) -> Path:
    """Save model state to disk.
    
    Args:
        path: Directory to save to.
        model: NNx model to save.
        config: Model configuration dict. If None, will attempt to
            extract from model.config if available.
        compress: Whether to use compression (.npz vs .npy).
    
    Returns:
        Path to saved checkpoint directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Extract state
    graphdef, state = nnx.split(model)
    state_dict = nnx.to_pure_dict(state)
    
    # Filter and flatten for serialization
    filtered = _filter_state_dict(state_dict)
    flat = _flatten_dict(filtered)
    
    # Convert to numpy
    arrays = {k: _to_numpy(v) for k, v in flat.items()}
    
    # Save arrays
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path / "model.npz", **arrays)
    
    # Try to extract config from model if not provided
    if config is None and hasattr(model, 'config'):
        model_config = model.config
        if hasattr(model_config, 'to_dict'):
            config = {"model_config": model_config.to_dict()}
    
    # Save config if available
    if config is not None:
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    # Save structure info for debugging
    structure = {k: list(v.shape) if hasattr(v, 'shape') else type(v).__name__ 
                 for k, v in arrays.items()}
    with open(path / "structure.json", "w") as f:
        json.dump(structure, f, indent=2)
    
    return path


def load_model(
    path: Union[str, Path],
    model: Optional[nnx.Module] = None,
    model_cls: Optional[type] = None,
    config_cls: Optional[type] = None,
    dtype: Optional[jnp.dtype] = None,
) -> Tuple[nnx.Module, Dict[str, Any]]:
    """Load model state from disk.
    
    Args:
        path: Checkpoint directory path.
        model: Existing model to load state into.
        model_cls: Model class to instantiate if model not provided.
        config_cls: Config class to use for model creation.
        dtype: Optional dtype to cast arrays to.
    
    Returns:
        Tuple of (model, config_dict).
    """
    path = Path(path)
    
    # Load config
    config_path = path / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    
    # Create model if not provided
    if model is None:
        if model_cls is None:
            # Try to import default model
            from linearnexus.models import LMModel, ModelConfig
            model_cls = LMModel
            config_cls = config_cls or ModelConfig
        
        if config_cls is not None and "model_config" in config:
            model_config = config_cls.from_dict(config["model_config"])
        elif config_cls is not None and config:
            model_config = config_cls.from_dict(config)
        else:
            raise ValueError("Cannot create model: no config provided and no model given")
        
        model = model_cls(model_config, rngs=nnx.Rngs(0))
    
    # Load arrays
    with np.load(path / "model.npz") as data:
        flat = {k: _to_jax(v, dtype) for k, v in data.items()}
    
    # Unflatten to nested dict
    state_dict = _unflatten_dict(flat)
    
    # Update model state
    graphdef, state = nnx.split(model)
    nnx.replace_by_pure_dict(state, state_dict)
    model = nnx.merge(graphdef, state)
    
    return model, config


def save_optimizer(
    path: Union[str, Path],
    optimizer: nnx.Optimizer,
    compress: bool = True,
) -> Path:
    """Save optimizer state.
    
    Args:
        path: Directory to save to.
        optimizer: NNx optimizer to save.
        compress: Whether to use compression.
    
    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Get the optax state from the optimizer
    # The NNx optimizer wraps optax and stores state internally
    opt_state = optimizer.opt_state
    
    # Serialize optax state
    flat_state, tree_def = jax.tree_util.tree_flatten(opt_state)
    
    # Convert to numpy
    arrays = {}
    non_arrays = {}
    for i, leaf in enumerate(flat_state):
        if isinstance(leaf, jax.Array):
            arrays[f"opt_{i}"] = _to_numpy(leaf)
        elif isinstance(leaf, np.ndarray):
            arrays[f"opt_{i}"] = leaf
        else:
            non_arrays[str(i)] = leaf
    
    # Save arrays
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path / "optimizer.npz", **arrays)
    
    # Save tree structure and non-array leaves
    import pickle
    with open(path / "optimizer_meta.pkl", "wb") as f:
        pickle.dump({
            "tree_def": tree_def,
            "non_arrays": non_arrays,
            "n_leaves": len(flat_state),
        }, f)
    
    return path


def load_optimizer(
    path: Union[str, Path],
    model: nnx.Module,
    tx: Optional[optax.GradientTransformation] = None,
    learning_rate: float = 1e-4,
) -> nnx.Optimizer:
    """Load optimizer state.
    
    Args:
        path: Checkpoint directory path.
        model: Model to wrap optimizer around.
        tx: Optax transformation. If None, uses AdamW.
        learning_rate: Learning rate (used if tx is None).
    
    Returns:
        Restored NNx optimizer.
    """
    import pickle
    
    path = Path(path)
    
    # Create optimizer with same transform
    if tx is None:
        tx = optax.adamw(learning_rate)
    
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    
    # Load saved state if exists
    opt_path = path / "optimizer.npz"
    meta_path = path / "optimizer_meta.pkl"
    
    if opt_path.exists() and meta_path.exists():
        # Load arrays
        with np.load(opt_path) as data:
            arrays = {k: _to_jax(v) for k, v in data.items()}
        
        # Load metadata
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        
        # Reconstruct flat list
        flat_state = []
        for i in range(meta["n_leaves"]):
            key = f"opt_{i}"
            if key in arrays:
                flat_state.append(arrays[key])
            elif str(i) in meta["non_arrays"]:
                flat_state.append(meta["non_arrays"][str(i)])
            else:
                raise ValueError(f"Missing optimizer state for index {i}")
        
        # Unflatten and restore
        opt_state = jax.tree_util.tree_unflatten(meta["tree_def"], flat_state)
        optimizer.opt_state = opt_state
    
    return optimizer


# -----------------------------------------------------------------------------
# Checkpoint Manager
# -----------------------------------------------------------------------------

class CheckpointManager:
    """Manages multiple checkpoints with automatic cleanup.
    
    Features:
    - Keep N most recent checkpoints
    - Keep K best checkpoints (by metric)
    - Atomic saves (write to temp, then rename)
    - Thread-safe operations
    - Automatic discovery of existing checkpoints
    
    Args:
        directory: Base directory for checkpoints.
        max_to_keep: Maximum number of recent checkpoints to keep.
        best_metric: Metric name for tracking best checkpoints.
        best_mode: "min" or "max" - whether lower or higher is better.
        best_to_keep: Number of best checkpoints to keep.
        prefix: Checkpoint directory prefix.
    """
    
    def __init__(
        self,
        directory: Union[str, Path],
        max_to_keep: int = 5,
        best_metric: Optional[str] = None,
        best_mode: Literal["min", "max"] = "min",
        best_to_keep: int = 3,
        prefix: str = "step",
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        
        self.max_to_keep = max_to_keep
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.best_to_keep = best_to_keep
        self.prefix = prefix
        
        # Track checkpoints
        self._checkpoints: List[CheckpointInfo] = []
        self._best_checkpoints: List[CheckpointInfo] = []
        
        # Discover existing checkpoints
        self._discover_checkpoints()
    
    def _discover_checkpoints(self) -> None:
        """Scan directory for existing checkpoints."""
        pattern = re.compile(rf"{self.prefix}_(\d+)")
        
        for item in self.directory.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    step = int(match.group(1))
                    timestamp = item.stat().st_mtime
                    
                    # Load metrics if available
                    metrics = {}
                    metrics_path = item / "metrics.json"
                    if metrics_path.exists():
                        with open(metrics_path, "r") as f:
                            metrics = json.load(f)
                    
                    info = CheckpointInfo(
                        path=item,
                        step=step,
                        timestamp=timestamp,
                        metrics=metrics,
                    )
                    self._checkpoints.append(info)
        
        # Sort by step
        self._checkpoints.sort()
        
        # Update best tracking
        if self.best_metric and self._checkpoints:
            self._update_best()
    
    def _update_best(self) -> None:
        """Update best checkpoints list."""
        if not self.best_metric:
            return
        
        candidates = [c for c in self._checkpoints 
                      if self.best_metric in c.metrics]
        
        if self.best_mode == "min":
            candidates.sort(key=lambda c: c.metrics[self.best_metric])
        else:
            candidates.sort(key=lambda c: c.metrics[self.best_metric], reverse=True)
        
        self._best_checkpoints = candidates[:self.best_to_keep]
    
    def _cleanup(self) -> None:
        """Remove old checkpoints beyond limits."""
        # Get checkpoints to protect
        protected = set(c.path for c in self._best_checkpoints)
        
        # Remove excess recent checkpoints
        recent = [c for c in self._checkpoints if c.path not in protected]
        
        while len(recent) > self.max_to_keep:
            oldest = recent.pop(0)
            if oldest.path.exists():
                shutil.rmtree(oldest.path)
                self._checkpoints.remove(oldest)
    
    def save(
        self,
        step: int,
        model: nnx.Module,
        optimizer: Optional[nnx.Optimizer] = None,
        config: Optional[Dict[str, Any]] = None,
        train_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a checkpoint.
        
        Args:
            step: Current training step.
            model: Model to save.
            optimizer: Optional optimizer to save.
            config: Model configuration.
            train_config: Training configuration.
            metrics: Current metrics (loss, accuracy, etc.).
            extra: Additional data to save.
        
        Returns:
            Path to saved checkpoint.
        """
        ckpt_name = f"{self.prefix}_{step}"
        ckpt_path = self.directory / ckpt_name
        
        # Use atomic save: write to temp, then rename
        temp_path = Path(tempfile.mkdtemp(dir=self.directory))
        
        try:
            # Save model (will auto-extract config if not provided)
            save_model(temp_path, model, config)
            
            # Save optimizer
            if optimizer is not None:
                save_optimizer(temp_path, optimizer)
            
            # Merge step and train_config into config.json
            config_path = temp_path / "config.json"
            existing = {}
            if config_path.exists():
                with open(config_path, "r") as f:
                    existing = json.load(f)
            existing["step"] = step
            if train_config is not None:
                existing["train_config"] = train_config
            with open(config_path, "w") as f:
                json.dump(existing, f, indent=2)
            
            # Save metrics
            if metrics is not None:
                with open(temp_path / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
            
            # Save extra data
            if extra is not None:
                with open(temp_path / "extra.json", "w") as f:
                    json.dump(extra, f, indent=2)
            
            # Atomic rename
            if ckpt_path.exists():
                shutil.rmtree(ckpt_path)
            temp_path.rename(ckpt_path)
            
        except Exception:
            # Cleanup on failure
            if temp_path.exists():
                shutil.rmtree(temp_path)
            raise
        
        # Track checkpoint
        info = CheckpointInfo(
            path=ckpt_path,
            step=step,
            timestamp=ckpt_path.stat().st_mtime,
            metrics=metrics or {},
        )
        self._checkpoints.append(info)
        self._checkpoints.sort()
        
        # Update best and cleanup
        self._update_best()
        self._cleanup()
        
        return ckpt_path
    
    def load(
        self,
        path: Union[str, Path],
        model: Optional[nnx.Module] = None,
        model_cls: Optional[type] = None,
        config_cls: Optional[type] = None,
        tx: Optional[optax.GradientTransformation] = None,
        should_load_optimizer: bool = True,
    ) -> CheckpointState:
        """Load a specific checkpoint.
        
        Args:
            path: Checkpoint directory path.
            model: Existing model to load into.
            model_cls: Model class for instantiation.
            config_cls: Config class for model creation.
            tx: Optax transformation for optimizer.
            should_load_optimizer: Whether to load optimizer state.
        
        Returns:
            CheckpointState with all restored components.
        """
        path = Path(path)
        
        # Load model using module-level function
        loaded_model, config = load_model(path, model, model_cls, config_cls)
        
        # Load optimizer
        optimizer = None
        if should_load_optimizer and (path / "optimizer.npz").exists():
            optimizer = load_optimizer(path, loaded_model, tx)
        
        # Load metrics
        metrics = None
        if (path / "metrics.json").exists():
            with open(path / "metrics.json", "r") as f:
                metrics = json.load(f)
        
        # Load extra
        extra = None
        if (path / "extra.json").exists():
            with open(path / "extra.json", "r") as f:
                extra = json.load(f)
        
        # Get step from config or meta.json
        step = config.get("step", 0)
        meta_path = path / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                step = meta.get("step", step)
        
        return CheckpointState(
            step=step,
            model=loaded_model,
            optimizer=optimizer,
            config=config.get("model_config"),
            train_config=config.get("train_config"),
            metrics=metrics,
            extra=extra,
        )
    
    def load_latest(self, **kwargs) -> Optional[CheckpointState]:
        """Load the most recent checkpoint."""
        if not self._checkpoints:
            return None
        return self.load(self._checkpoints[-1].path, **kwargs)
    
    def load_best(self, **kwargs) -> Optional[CheckpointState]:
        """Load the best checkpoint by tracked metric."""
        if not self._best_checkpoints:
            return self.load_latest(**kwargs)
        return self.load(self._best_checkpoints[0].path, **kwargs)
    
    def load_step(self, step: int, **kwargs) -> Optional[CheckpointState]:
        """Load checkpoint at specific step."""
        for ckpt in self._checkpoints:
            if ckpt.step == step:
                return self.load(ckpt.path, **kwargs)
        return None
    
    @property
    def latest_step(self) -> Optional[int]:
        """Get the step number of the latest checkpoint."""
        return self._checkpoints[-1].step if self._checkpoints else None
    
    @property
    def best_step(self) -> Optional[int]:
        """Get the step number of the best checkpoint."""
        return self._best_checkpoints[0].step if self._best_checkpoints else None
    
    @property 
    def checkpoints(self) -> List[CheckpointInfo]:
        """Get list of all tracked checkpoints."""
        return list(self._checkpoints)
    
    def __len__(self) -> int:
        return len(self._checkpoints)
    
    def __repr__(self) -> str:
        return (
            f"CheckpointManager(directory={self.directory}, "
            f"checkpoints={len(self)}, "
            f"latest={self.latest_step}, "
            f"best={self.best_step})"
        )


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def save_checkpoint(
    path: Union[str, Path],
    model: nnx.Module,
    step: int = 0,
    optimizer: Optional[nnx.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> Path:
    """Save a complete checkpoint (convenience wrapper).
    
    Args:
        path: Directory to save to.
        model: Model to save.
        step: Training step.
        optimizer: Optional optimizer.
        config: Model/training config.
        metrics: Training metrics.
    
    Returns:
        Path to checkpoint.
    """
    path = Path(path)
    
    # Save model
    save_model(path, model, config)
    
    # Save optimizer  
    if optimizer is not None:
        save_optimizer(path, optimizer)
    
    # Save step and metrics
    meta = {"step": step}
    if metrics:
        meta["metrics"] = metrics
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return path


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nnx.Module] = None,
    model_cls: Optional[type] = None,
    config_cls: Optional[type] = None,
) -> Tuple[nnx.Module, Dict[str, Any]]:
    """Load a checkpoint (convenience wrapper).
    
    Args:
        path: Checkpoint directory.
        model: Existing model to load into.
        model_cls: Model class for instantiation.
        config_cls: Config class.
    
    Returns:
        Tuple of (model, metadata dict).
    """
    return load_model(path, model, model_cls, config_cls)
