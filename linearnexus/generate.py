"""Text generation utilities for LLM inference.

Provides unified generation interface for all model architectures:
- Autoregressive token-by-token generation
- Sampling strategies (greedy, temperature, top-k, top-p)
- Batched generation for throughput
- State caching (KV cache for attention, SSM state for Mamba)

Example:
    model = LMModel(config, rngs=nnx.Rngs(0))
    prompt = tokenizer.encode("Once upon a time")
    
    output = generate(
        model,
        jnp.array([prompt]),
        max_tokens=100,
        temperature=0.8,
        top_k=50,
    )
    
    text = tokenizer.decode(output[0].tolist())
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.models import LMModel, ModelState

Array = jax.Array


# -----------------------------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------------------------

def sample_top_k(
    logits: Array,
    k: int,
    temperature: float = 1.0,
    key: Array = None,
) -> Array:
    """Sample from top-k logits.
    
    Args:
        logits: Logits array [batch, vocab_size].
        k: Number of top tokens to consider.
        temperature: Sampling temperature (1.0 = no change).
        key: JAX random key.
        
    Returns:
        Sampled token IDs [batch].
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Apply temperature
    logits = logits / temperature
    
    # Get top-k
    top_k_logits, top_k_indices = jax.lax.top_k(logits, k)
    
    # Sample from top-k
    probs = jax.nn.softmax(top_k_logits, axis=-1)
    sampled_indices = jax.random.categorical(key, jnp.log(probs + 1e-10), axis=-1)
    
    # Map back to vocabulary indices
    batch_size = logits.shape[0]
    tokens = top_k_indices[jnp.arange(batch_size), sampled_indices]
    
    return tokens


def sample_top_p(
    logits: Array,
    p: float,
    temperature: float = 1.0,
    key: Array = None,
) -> Array:
    """Sample from top-p (nucleus) logits.
    
    Args:
        logits: Logits array [batch, vocab_size].
        p: Cumulative probability threshold (0.0-1.0).
        temperature: Sampling temperature.
        key: JAX random key.
        
    Returns:
        Sampled token IDs [batch].
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Apply temperature
    logits = logits / temperature
    
    # Sort by probability
    sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
    
    # Compute cumulative probabilities
    probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumsum_probs = jnp.cumsum(probs, axis=-1)
    
    # Create mask for tokens within top-p
    mask = cumsum_probs <= p
    # Always include at least one token
    mask = mask.at[:, 0].set(True)
    
    # Mask out tokens outside top-p
    masked_logits = jnp.where(mask, sorted_logits, -1e10)
    
    # Sample
    sampled_indices = jax.random.categorical(key, masked_logits, axis=-1)
    
    # Map back to vocabulary indices
    batch_size = logits.shape[0]
    tokens = sorted_indices[jnp.arange(batch_size), sampled_indices]
    
    return tokens


def sample_token(
    logits: Array,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    key: Array = None,
) -> Array:
    """Sample next token from logits.
    
    Applies sampling strategy in order:
    1. Temperature scaling
    2. Top-k filtering (if specified)
    3. Top-p filtering (if specified)
    4. Categorical sampling (or argmax if temperature=0)
    
    Args:
        logits: Logits for next token [batch, vocab_size].
        temperature: Sampling temperature. 0 = greedy.
        top_k: If set, sample from top k tokens.
        top_p: If set, sample from nucleus (top-p).
        key: JAX random key.
        
    Returns:
        Sampled token IDs [batch].
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Greedy decoding
    if temperature == 0:
        return jnp.argmax(logits, axis=-1)
    
    # Temperature scaling
    logits = logits / temperature
    
    # Top-k filtering
    if top_k is not None and top_k > 0:
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        # Create mask for non-top-k tokens
        mask = jnp.ones_like(logits, dtype=jnp.bool_)
        mask = mask.at[jnp.arange(logits.shape[0])[:, None], top_k_indices].set(False)
        logits = jnp.where(mask, -1e10, logits)
    
    # Top-p filtering
    if top_p is not None and top_p < 1.0:
        sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumsum = jnp.cumsum(probs, axis=-1)
        
        # Mask tokens beyond threshold
        mask = cumsum > top_p
        mask = mask.at[:, 0].set(False)  # Keep at least one token
        sorted_logits = jnp.where(mask, -1e10, sorted_logits)
        
        # Unsort
        unsort_indices = jnp.argsort(sorted_indices, axis=-1)
        logits = jnp.take_along_axis(sorted_logits, unsort_indices, axis=-1)
    
    # Sample from distribution
    return jax.random.categorical(key, logits, axis=-1)


# -----------------------------------------------------------------------------
# Generation Loop
# -----------------------------------------------------------------------------

def generate(
    model: LMModel,
    prompt_tokens: Array,
    max_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    key: Optional[Array] = None,
) -> Array:
    """Generate text autoregressively.
    
    Args:
        model: Language model (LMModel instance).
        prompt_tokens: Prompt token IDs [batch, prompt_len].
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        eos_token_id: Stop generation at this token.
        pad_token_id: Token ID for padding shorter sequences.
        key: JAX random key for sampling.
        
    Returns:
        Generated token IDs [batch, prompt_len + max_tokens].
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    batch_size, prompt_len = prompt_tokens.shape
    dtype = prompt_tokens.dtype
    
    # Process prompt to get initial state
    logits, state = model(prompt_tokens, mode="chunk")
    
    # Initialize output buffer
    total_len = prompt_len + max_tokens
    output = jnp.zeros((batch_size, total_len), dtype=dtype)
    output = output.at[:, :prompt_len].set(prompt_tokens)
    
    # Track which sequences have finished (hit EOS)
    finished = jnp.zeros(batch_size, dtype=jnp.bool_)
    
    # Get last token logits
    next_logits = logits[:, -1, :]  # [batch, vocab]
    
    # Generate tokens one at a time
    for i in range(max_tokens):
        key, subkey = jax.random.split(key)
        
        # Sample next token
        next_token = sample_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            key=subkey,
        )
        
        # Replace with pad token for finished sequences
        if pad_token_id is not None:
            next_token = jnp.where(finished, pad_token_id, next_token)
        
        # Store token
        output = output.at[:, prompt_len + i].set(next_token)
        
        # Check for EOS
        if eos_token_id is not None:
            finished = finished | (next_token == eos_token_id)
            if jnp.all(finished):
                break
        
        # Forward pass for next token (recurrent mode for efficiency)
        next_token_input = next_token[:, None]  # [batch, 1]
        next_logits, state = model(next_token_input, state=state, mode="recurrent")
        next_logits = next_logits[:, -1, :]  # [batch, vocab]
    
    return output


def generate_streaming(
    model: LMModel,
    prompt_tokens: Array,
    max_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    key: Optional[Array] = None,
):
    """Generate text with streaming output (yields token by token).
    
    Yields:
        Tuple of (token_id, token_position) for each generated token.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    batch_size, prompt_len = prompt_tokens.shape
    
    # Process prompt
    logits, state = model(prompt_tokens, mode="chunk")
    next_logits = logits[:, -1, :]
    
    for i in range(max_tokens):
        key, subkey = jax.random.split(key)
        
        # Sample
        next_token = sample_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            key=subkey,
        )
        
        yield next_token, prompt_len + i
        
        # Check EOS
        if eos_token_id is not None and jnp.all(next_token == eos_token_id):
            break
        
        # Next step
        next_token_input = next_token[:, None]
        next_logits, state = model(next_token_input, state=state, mode="recurrent")
        next_logits = next_logits[:, -1, :]


# -----------------------------------------------------------------------------
# Batch Generation Utilities
# -----------------------------------------------------------------------------

def batch_generate(
    model: LMModel,
    prompts: List[List[int]],
    max_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: int = 0,
    key: Optional[Array] = None,
) -> List[List[int]]:
    """Generate from multiple prompts with different lengths.
    
    Pads prompts to same length, generates, then strips padding.
    
    Args:
        model: Language model.
        prompts: List of token ID lists (variable lengths).
        max_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature.
        top_k: Top-k parameter.
        top_p: Top-p parameter.
        eos_token_id: EOS token ID.
        pad_token_id: Padding token ID.
        key: Random key.
        
    Returns:
        List of generated token ID lists.
    """
    if not prompts:
        return []
    
    # Pad prompts to same length
    max_prompt_len = max(len(p) for p in prompts)
    batch_size = len(prompts)
    
    padded = jnp.full((batch_size, max_prompt_len), pad_token_id, dtype=jnp.int32)
    prompt_lens = []
    
    for i, prompt in enumerate(prompts):
        prompt_len = len(prompt)
        prompt_lens.append(prompt_len)
        padded = padded.at[i, :prompt_len].set(jnp.array(prompt))
    
    # Generate
    output = generate(
        model,
        padded,
        max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        key=key,
    )
    
    # Extract results (strip left padding, optionally trim at EOS)
    results = []
    for i in range(batch_size):
        seq = output[i].tolist()
        
        # Find actual content (skip padding)
        start = max_prompt_len - prompt_lens[i]
        seq = seq[start:]
        
        # Trim at EOS if present
        if eos_token_id is not None and eos_token_id in seq:
            eos_idx = seq.index(eos_token_id)
            seq = seq[:eos_idx + 1]
        
        results.append(seq)
    
    return results


# -----------------------------------------------------------------------------
# Chat/Completion Helpers
# -----------------------------------------------------------------------------

def complete(
    model: LMModel,
    tokenizer,  # Any tokenizer with encode/decode
    prompt: str,
    max_tokens: int = 100,
    *,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
    key: Optional[Array] = None,
) -> str:
    """Simple text completion helper.
    
    Args:
        model: Language model.
        tokenizer: Tokenizer with encode/decode methods.
        prompt: Text prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k parameter.
        top_p: Top-p parameter.
        key: Random key.
        
    Returns:
        Generated text (including prompt).
    """
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_array = jnp.array([prompt_tokens], dtype=jnp.int32)
    
    # Generate
    eos_id = getattr(tokenizer, "eos_token_id", None)
    output = generate(
        model,
        prompt_array,
        max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_id,
        key=key,
    )
    
    # Decode
    return tokenizer.decode(output[0].tolist())
