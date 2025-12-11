#!/usr/bin/env python3
"""Generate text from trained language models.

nanoGPT-style sampling script for LinearNexus.

Examples:
    # Generate from checkpoint
    python sample.py --checkpoint checkpoints/step_5000 --prompt "To be or not to be"
    
    # Greedy decoding
    python sample.py --checkpoint checkpoints/step_5000 --temperature 0
    
    # Creative sampling with top-k
    python sample.py --checkpoint checkpoints/step_5000 --temperature 1.2 --top-k 100
    
    # Generate multiple samples
    python sample.py --checkpoint checkpoints/step_5000 --num-samples 3
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx

sys.path.insert(0, str(Path(__file__).parent))

from linearnexus.models import LMModel, ModelConfig
from linearnexus.data import CharTokenizer
from linearnexus.generate import generate, complete


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer (auto-detected if not specified)")
    
    # Generation
    parser.add_argument("--prompt", type=str, default="", help="Text prompt to start generation")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    
    # Sampling
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode (enter prompts)")
    
    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: Path):
    """Load model and tokenizer from checkpoint."""
    
    # Load config
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path) as f:
        meta = json.load(f)
    
    model_config = ModelConfig.from_dict(meta["model_config"])
    
    # Create model
    rngs = nnx.Rngs(0)
    model = LMModel(model_config, rngs=rngs)
    
    # Load state
    state_path = checkpoint_path / "model_state.json"
    if state_path.exists():
        with open(state_path) as f:
            state_dict = json.load(f)
        
        # Convert to arrays (simplified)
        def to_array(x):
            if isinstance(x, list):
                return jnp.array(x)
            return x
        
        state_dict = jax.tree.map(to_array, state_dict)
        # Would update model params here in full implementation
    
    # Load tokenizer
    # Try to find tokenizer or vocab file
    tokenizer_path = checkpoint_path / "tokenizer.pkl"
    vocab_path = checkpoint_path / "vocab.txt"
    
    if tokenizer_path.exists():
        tokenizer = CharTokenizer.load(tokenizer_path)
    elif vocab_path.exists():
        with open(vocab_path) as f:
            chars = f.read()
        tokenizer = CharTokenizer(chars)
    else:
        # Create a basic tokenizer from ASCII
        print("Warning: No tokenizer found, using default ASCII characters")
        chars = "".join(chr(i) for i in range(32, 127)) + "\n\t"
        tokenizer = CharTokenizer(chars)
    
    return model, tokenizer


def generate_text(
    model: LMModel,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    key: jax.Array,
) -> str:
    """Generate text continuation."""
    
    # Encode prompt
    if prompt:
        prompt_tokens = tokenizer.encode(prompt)
    else:
        # Start with newline if no prompt
        prompt_tokens = [0]  # First token
    
    prompt_array = jnp.array([prompt_tokens], dtype=jnp.int32)
    
    # Generate
    output = generate(
        model,
        prompt_array,
        max_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        top_p=top_p,
        key=key,
    )
    
    # Decode
    return tokenizer.decode(output[0].tolist())


def interactive_mode(model, tokenizer, args):
    """Interactive generation mode."""
    print("\nInteractive mode. Type 'quit' or 'exit' to stop.")
    print("-" * 40)
    
    key = jax.random.PRNGKey(args.seed)
    
    while True:
        try:
            prompt = input("\nPrompt> ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        if not prompt:
            continue
        
        key, subkey = jax.random.split(key)
        
        print("\nGenerating...\n")
        output = generate_text(
            model, tokenizer, prompt,
            args.max_tokens, args.temperature,
            args.top_k, args.top_p, subkey,
        )
        print(output)


def main():
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print(f"Loading model from {checkpoint_path}...")
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    if args.interactive:
        interactive_mode(model, tokenizer, args)
        return
    
    # Generate samples
    key = jax.random.PRNGKey(args.seed)
    
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print("-" * 60)
    
    for i in range(args.num_samples):
        key, subkey = jax.random.split(key)
        
        output = generate_text(
            model, tokenizer, args.prompt,
            args.max_tokens, args.temperature,
            args.top_k, args.top_p, subkey,
        )
        
        if args.num_samples > 1:
            print(f"\n--- Sample {i + 1} ---")
        print(output)
    
    print("-" * 60)


if __name__ == "__main__":
    main()
