"""Custom optimizers for LLM training.

Implements modern optimizers beyond standard Adam:
- AdamW: Decoupled weight decay (standard baseline)
- Muon: Momentum orthogonalization for stable training
- Sophia: Second-order Hessian-based optimizer

All optimizers follow the optax.GradientTransformation interface,
enabling composition with gradient clipping, learning rate schedules, etc.

Example:
    optimizer = get_optimizer("muon", lr=1e-3, momentum=0.9)
    
    # Compose with gradient clipping
    import optax
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        get_optimizer("adamw", lr=3e-4, weight_decay=0.1),
    )
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from optax import GradientTransformation, Updates, Params, OptState

Array = jax.Array
Scalar = Union[float, Array]


# -----------------------------------------------------------------------------
# AdamW (Wrapped Optax with Sensible Defaults)
# -----------------------------------------------------------------------------

def adamw(
    learning_rate: Scalar = 1e-4,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    mask: Optional[Callable[[Params], Any]] = None,
) -> GradientTransformation:
    """AdamW optimizer with decoupled weight decay.
    
    Uses LLM-friendly defaults (b2=0.95 for stability).
    
    Args:
        learning_rate: Learning rate (scalar or schedule).
        b1: Exponential decay rate for first moment.
        b2: Exponential decay rate for second moment.
        eps: Small constant for numerical stability.
        weight_decay: Weight decay coefficient.
        mask: Optional mask function for weight decay (e.g., exclude biases).
        
    Returns:
        Optax GradientTransformation.
    """
    return optax.adamw(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        eps=eps,
        weight_decay=weight_decay,
        mask=mask,
    )


# -----------------------------------------------------------------------------
# Muon (Momentum Orthogonalization)
# -----------------------------------------------------------------------------

class MuonState(NamedTuple):
    """State for Muon optimizer."""
    momentum: Updates  # Momentum buffer
    count: Array       # Step count


def muon(
    learning_rate: Scalar = 1e-3,
    momentum: float = 0.95,
    nesterov: bool = True,
    backend_steps: int = 5,
) -> GradientTransformation:
    """Muon optimizer: Momentum with orthogonalization.
    
    Muon (Momentum Orthogonalization) applies Newton-Schulz orthogonalization
    to the momentum buffer, which can improve training stability and enable
    larger learning rates.
    
    Based on: https://kellerjordan.github.io/posts/muon/
    
    The key insight is that orthogonalizing the update direction prevents
    the optimizer from collapsing dimensions and enables more aggressive updates.
    
    Args:
        learning_rate: Learning rate (scalar or schedule).
        momentum: Momentum coefficient (typically 0.95).
        nesterov: Whether to use Nesterov momentum.
        backend_steps: Number of Newton-Schulz iterations for orthogonalization.
        
    Returns:
        Optax GradientTransformation.
    """
    
    def init_fn(params: Params) -> MuonState:
        return MuonState(
            momentum=jax.tree.map(jnp.zeros_like, params),
            count=jnp.array(0, dtype=jnp.int32),
        )
    
    def update_fn(
        updates: Updates,
        state: MuonState,
        params: Optional[Params] = None,
    ) -> Tuple[Updates, MuonState]:
        del params  # Unused
        
        # Get learning rate (handle schedules)
        lr = learning_rate
        if callable(learning_rate):
            lr = learning_rate(state.count)
        
        # Update momentum: m = momentum * m + (1 - momentum) * g
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + (1 - momentum) * g,
            state.momentum, updates
        )
        
        # Apply orthogonalization to momentum
        def orthogonalize(m: Array) -> Array:
            """Newton-Schulz orthogonalization for matrices, identity for vectors."""
            if m.ndim < 2:
                return m
            
            # Reshape to 2D for orthogonalization
            original_shape = m.shape
            if m.ndim > 2:
                m = m.reshape(m.shape[0], -1)
            
            # Scale for numerical stability
            scale = jnp.sqrt(jnp.mean(m ** 2) + 1e-8)
            m_scaled = m / scale
            
            # Newton-Schulz iterations: X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k^T @ X_k
            x = m_scaled
            for _ in range(backend_steps):
                x = 1.5 * x - 0.5 * x @ (x.T @ x)
            
            # Rescale
            result = x * scale
            
            # Reshape back
            if len(original_shape) > 2:
                result = result.reshape(original_shape)
            
            return result
        
        # Apply orthogonalization
        ortho_momentum = jax.tree.map(orthogonalize, new_momentum)
        
        # Compute update
        if nesterov:
            # Nesterov: use lookahead momentum
            updates_out = jax.tree.map(
                lambda m, om: -lr * (momentum * om + (1 - momentum) * m),
                updates, ortho_momentum
            )
        else:
            updates_out = jax.tree.map(lambda om: -lr * om, ortho_momentum)
        
        new_state = MuonState(
            momentum=new_momentum,
            count=state.count + 1,
        )
        
        return updates_out, new_state
    
    return GradientTransformation(init_fn, update_fn)


# -----------------------------------------------------------------------------
# Sophia (Second-Order with Hutchinson Estimator)
# -----------------------------------------------------------------------------

class SophiaState(NamedTuple):
    """State for Sophia optimizer."""
    exp_avg: Updates       # First moment (like Adam m)
    hessian_diag: Updates  # Diagonal Hessian estimate
    count: Array           # Step count
    rng_key: Array         # For Hutchinson sampling


def sophia(
    learning_rate: Scalar = 1e-4,
    b1: float = 0.9,
    b2: float = 0.99,
    rho: float = 0.04,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    hessian_update_freq: int = 10,
    seed: int = 0,
) -> GradientTransformation:
    """Sophia optimizer: Second-order with Hutchinson Hessian estimation.
    
    Sophia uses diagonal Hessian estimates via the Hutchinson trace estimator
    to adapt learning rates per-parameter. This can lead to faster convergence
    than AdamW on some tasks.
    
    Based on: https://arxiv.org/abs/2305.14342
    
    Note: This implementation requires the loss function to be available
    for Hessian computation. For standard training, you'll need to pass
    the loss function separately or use the simplified diagonal approximation.
    
    This simplified version uses the EMA of squared gradients as a Hessian
    proxy (similar to RMSprop/Adam), which is a reasonable approximation.
    
    Args:
        learning_rate: Learning rate.
        b1: Exponential decay for gradient EMA.
        b2: Exponential decay for Hessian diagonal EMA.
        rho: Clipping threshold for Hessian-scaled updates.
        eps: Numerical stability constant.
        weight_decay: Weight decay coefficient.
        hessian_update_freq: How often to update Hessian estimate (unused in simplified version).
        seed: Random seed for Hutchinson vectors.
        
    Returns:
        Optax GradientTransformation.
    """
    
    def init_fn(params: Params) -> SophiaState:
        return SophiaState(
            exp_avg=jax.tree.map(jnp.zeros_like, params),
            hessian_diag=jax.tree.map(jnp.zeros_like, params),
            count=jnp.array(0, dtype=jnp.int32),
            rng_key=jax.random.PRNGKey(seed),
        )
    
    def update_fn(
        updates: Updates,
        state: SophiaState,
        params: Optional[Params] = None,
    ) -> Tuple[Updates, SophiaState]:
        # Get learning rate
        lr = learning_rate
        if callable(learning_rate):
            lr = learning_rate(state.count)
        
        count = state.count + 1
        
        # Update first moment (gradient EMA)
        new_exp_avg = jax.tree.map(
            lambda m, g: b1 * m + (1 - b1) * g,
            state.exp_avg, updates
        )
        
        # Update Hessian diagonal estimate
        # Simplified: use squared gradients as proxy (like RMSprop)
        new_hessian = jax.tree.map(
            lambda h, g: b2 * h + (1 - b2) * (g ** 2),
            state.hessian_diag, updates
        )
        
        # Bias correction
        bias_correction1 = 1 - b1 ** count
        bias_correction2 = 1 - b2 ** count
        
        # Compute Sophia update with clipping
        def sophia_update(m, h, p):
            m_hat = m / bias_correction1
            h_hat = h / bias_correction2
            
            # Clip update: min(|m / (h + eps)|, rho) * sign(m)
            scaled = m_hat / (h_hat + eps)
            clipped = jnp.clip(scaled, -rho, rho)
            
            update = -lr * clipped
            
            # Weight decay
            if weight_decay > 0 and p is not None:
                update = update - lr * weight_decay * p
            
            return update
        
        if params is not None:
            updates_out = jax.tree.map(sophia_update, new_exp_avg, new_hessian, params)
        else:
            updates_out = jax.tree.map(
                lambda m, h: sophia_update(m, h, None),
                new_exp_avg, new_hessian
            )
        
        new_state = SophiaState(
            exp_avg=new_exp_avg,
            hessian_diag=new_hessian,
            count=count,
            rng_key=state.rng_key,  # Not used in simplified version
        )
        
        return updates_out, new_state
    
    return GradientTransformation(init_fn, update_fn)


# -----------------------------------------------------------------------------
# Learning Rate Schedules
# -----------------------------------------------------------------------------

def cosine_schedule(
    init_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
) -> Callable[[int], float]:
    """Cosine learning rate schedule with warmup.
    
    Args:
        init_lr: Peak learning rate (reached after warmup).
        total_steps: Total training steps.
        warmup_steps: Number of warmup steps.
        min_lr: Minimum learning rate at end.
        
    Returns:
        Schedule function: step -> learning_rate.
    """
    def schedule(step: int) -> float:
        step = jnp.asarray(step, dtype=jnp.float32)
        
        # Warmup phase
        warmup_lr = init_lr * step / max(warmup_steps, 1)
        
        # Cosine decay phase
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = jnp.clip(progress, 0.0, 1.0)
        cosine_lr = min_lr + (init_lr - min_lr) * 0.5 * (1 + jnp.cos(jnp.pi * progress))
        
        return jnp.where(step < warmup_steps, warmup_lr, cosine_lr)
    
    return schedule


def linear_warmup_schedule(
    init_lr: float,
    warmup_steps: int,
) -> Callable[[int], float]:
    """Linear warmup to target learning rate.
    
    Args:
        init_lr: Target learning rate.
        warmup_steps: Number of warmup steps.
        
    Returns:
        Schedule function.
    """
    def schedule(step: int) -> float:
        step = jnp.asarray(step, dtype=jnp.float32)
        return jnp.minimum(init_lr, init_lr * step / max(warmup_steps, 1))
    
    return schedule


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------

def get_optimizer(
    name: str,
    learning_rate: Scalar = 1e-4,
    **kwargs,
) -> GradientTransformation:
    """Get optimizer by name.
    
    Args:
        name: Optimizer name ("adamw", "muon", "sophia", "sgd").
        learning_rate: Learning rate (scalar or schedule).
        **kwargs: Optimizer-specific parameters.
        
    Returns:
        Optax GradientTransformation.
        
    Example:
        optimizer = get_optimizer("adamw", learning_rate=3e-4, weight_decay=0.1)
        optimizer = get_optimizer("muon", learning_rate=1e-3, momentum=0.95)
    """
    name = name.lower()
    
    if name == "adamw":
        return adamw(learning_rate=learning_rate, **kwargs)
    
    elif name == "muon":
        return muon(learning_rate=learning_rate, **kwargs)
    
    elif name == "sophia":
        return sophia(learning_rate=learning_rate, **kwargs)
    
    elif name == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        nesterov = kwargs.get("nesterov", True)
        return optax.sgd(learning_rate, momentum=momentum, nesterov=nesterov)
    
    elif name == "adam":
        b1 = kwargs.get("b1", 0.9)
        b2 = kwargs.get("b2", 0.999)
        eps = kwargs.get("eps", 1e-8)
        return optax.adam(learning_rate, b1=b1, b2=b2, eps=eps)
    
    else:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Choose from: 'adamw', 'muon', 'sophia', 'sgd', 'adam'"
        )


def create_optimizer(
    name: str,
    learning_rate: float,
    total_steps: Optional[int] = None,
    warmup_steps: int = 0,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    **kwargs,
) -> GradientTransformation:
    """Create optimizer with common training setup.

    Combines optimizer with:
    - Gradient clipping
    - Cosine learning rate schedule with warmup
    - Weight decay (for AdamW/Sophia)

    Args:
        name: Optimizer name.
        learning_rate: Peak learning rate.
        total_steps: Total training steps. If `None`, defaults to 100000.
        warmup_steps: Warmup steps.
        weight_decay: Weight decay (passed to optimizer).
        grad_clip: Gradient clipping norm.
        **kwargs: Additional optimizer parameters.

    Returns:
        Composed GradientTransformation.
    """
    # Provide a sensible default when total_steps is not supplied.
    if total_steps is None:
        total_steps = 100_000

    # Create learning rate schedule
    schedule = cosine_schedule(
        init_lr=learning_rate,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=learning_rate * 0.1,  # Decay to 10% of peak
    )
    
    # Build optimizer chain
    transforms = []
    
    # Gradient clipping
    if grad_clip > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip))
    
    # Main optimizer
    if name in ("adamw", "sophia"):
        kwargs["weight_decay"] = weight_decay
    
    transforms.append(get_optimizer(name, learning_rate=schedule, **kwargs))
    
    return optax.chain(*transforms)
