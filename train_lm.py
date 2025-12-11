#!/usr/bin/env python3
"""Train language models on text data.

nanoGPT-style training script for LinearNexus.

Examples:
    # Train GPT on Shakespeare
    python train_lm.py --model gpt-small --data data/shakespeare.txt
    
    # Train Mamba with Muon optimizer
    python train_lm.py --model mamba-small --optimizer muon --lr 1e-3
    
    # Train hybrid Jamba-style model
    python train_lm.py --model jamba-small --batch-size 2
    
    # Resume from checkpoint
    python train_lm.py --model gpt-small --resume checkpoints/step_5000
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from linearnexus.models import create_model, LMModel, ModelConfig
from linearnexus.data import CharTokenizer, TextDataset, DataLoader, download_shakespeare
from linearnexus.optim import create_optimizer, get_optimizer
from linearnexus.train import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model
    parser.add_argument(
        "--model", type=str, default="gpt-small",
        choices=["gpt-small", "gpt-medium", "mamba-small", "mamba-medium", "jamba-small", "custom"],
        help="Model architecture preset",
    )
    parser.add_argument("--hidden-size", type=int, default=None, help="Override hidden size")
    parser.add_argument("--n-layers", type=int, default=None, help="Override number of layers")
    parser.add_argument("--n-heads", type=int, default=None, help="Override attention heads")
    
    # Data
    parser.add_argument("--data", type=str, default=None, help="Path to training text file")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--download-shakespeare", action="store_true", help="Download tiny Shakespeare")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    
    # Optimizer
    parser.add_argument(
        "--optimizer", type=str, default="adamw",
        choices=["adamw", "muon", "sophia", "sgd"],
        help="Optimizer type",
    )
    
    # Checkpointing
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save checkpoint interval")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compile", action="store_true", help="JIT compile training step")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LinearNexus Training Script")
    print("=" * 60)
    
    # Set random seed
    key = jax.random.PRNGKey(args.seed)
    
    # Download data if needed
    if args.download_shakespeare or args.data is None:
        data_path = download_shakespeare()
    else:
        data_path = Path(args.data)
    
    print(f"\nData: {data_path}")
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = CharTokenizer.from_file(data_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset and dataloader
    print(f"Loading dataset with seq_len={args.seq_len}...")
    dataset = TextDataset(data_path, tokenizer, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    print(f"Dataset size: {len(dataset)} samples, {len(dataloader)} batches per epoch")
    
    # Build model config with overrides
    config_overrides = {"vocab_size": tokenizer.vocab_size}
    if args.hidden_size:
        config_overrides["hidden_size"] = args.hidden_size
    if args.n_layers:
        config_overrides["n_layers"] = args.n_layers
    if args.n_heads:
        config_overrides["n_heads"] = args.n_heads
    
    config, _ = create_model(args.model, **config_overrides)
    
    print(f"\nModel: {args.model}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  block_pattern: {config.block_pattern}")
    if "attention" in config.block_pattern:
        print(f"  n_heads: {config.n_heads}")
    if "mamba" in config.block_pattern:
        print(f"  state_size: {config.state_size}")
    
    # Create model
    print("\nInitializing model...")
    rngs = nnx.Rngs(args.seed)
    model = LMModel(config, rngs=rngs)
    
    n_params = model.count_params()
    print(f"Total parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
    
    # Create optimizer
    print(f"\nOptimizer: {args.optimizer}")
    optimizer = create_optimizer(
        args.optimizer,
        learning_rate=args.lr,
        total_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )
    
    # Create training config
    train_config = SFTConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    # Create trainer
    trainer = SFTTrainer(model, optimizer, train_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        # Would load checkpoint here
    
    # Train!
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.train(iter(dataloader))
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
