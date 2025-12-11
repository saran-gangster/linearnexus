"""LinearNexus: Minimal LLM training framework in JAX.

A nanoGPT-style implementation supporting multiple architectures:
- Dense attention (GPT-style)
- State-space models (Mamba)
- Hybrid patterns (Jamba-style interleaved)

Plus complete training infrastructure:
- Custom optimizers (AdamW, Muon, Sophia)
- Training modes (SFT, GRPO, PPO)
- Data loading and tokenization
- Text generation

Example:
    from linearnexus import LMModel, ModelConfig, create_model
    from linearnexus import CharTokenizer, TextDataset, DataLoader
    from linearnexus import create_optimizer, SFTTrainer, SFTConfig
    from linearnexus import generate, complete
    
    # Create GPT model
    config, _ = create_model("gpt-small", vocab_size=256)
    model = LMModel(config, rngs=nnx.Rngs(0))
    
    # Train
    trainer = SFTTrainer(model, optimizer, train_config)
    trainer.train(dataloader)
    
    # Generate
    output = generate(model, prompt_tokens, max_tokens=100)
"""

from linearnexus.models import (
    LMModel,
    ModelConfig,
    ModelState,
    create_model,
    GPT_SMALL,
    GPT_MEDIUM,
    MAMBA_SMALL,
    MAMBA_MEDIUM,
    JAMBA_SMALL,
    MAMBA2_SMALL,
    MAMBA2_MEDIUM,
    JAMBA2_SMALL,
)

from linearnexus.data import (
    CharTokenizer,
    BPETokenizer,
    TextDataset,
    DataLoader,
    download_shakespeare,
)

from linearnexus.optim import (
    adamw,
    muon,
    sophia,
    get_optimizer,
    create_optimizer,
    cosine_schedule,
)

from linearnexus.train import (
    SFTTrainer,
    SFTConfig,
    GRPOTrainer,
    GRPOConfig,
    PPOTrainer,
    PPOConfig,
    cross_entropy_loss,
)

# Checkpointing
from linearnexus.checkpoint import (
    CheckpointManager,
    CheckpointState,
    CheckpointInfo,
    save_checkpoint,
    load_checkpoint,
    save_model,
    load_model,
    save_optimizer,
    load_optimizer,
)

from linearnexus.generate import (
    generate,
    generate_streaming,
    complete,
    batch_generate,
    sample_token,
)

# Module-level exports
from linearnexus.modules.attention import AttentionBlock, CausalSelfAttention, KVCache
from linearnexus.modules.ssm import MambaBlock, MambaState
from linearnexus.modules.common import MLP, RMSNorm, Embedding, RotaryEmbedding

__version__ = "0.2.0"

__all__ = [
    # Models
    "LMModel",
    "ModelConfig", 
    "ModelState",
    "create_model",
    "GPT_SMALL",
    "GPT_MEDIUM",
    "MAMBA_SMALL",
    "MAMBA_MEDIUM",
    "JAMBA_SMALL",
    
    # Data
    "CharTokenizer",
    "BPETokenizer",
    "TextDataset",
    "DataLoader",
    "download_shakespeare",
    
    # Optimizers
    "adamw",
    "muon",
    "sophia",
    "get_optimizer",
    "create_optimizer",
    "cosine_schedule",
    
    # Training
    "SFTTrainer",
    "SFTConfig",
    "GRPOTrainer",
    "GRPOConfig",
    "PPOTrainer",
    "PPOConfig",
    "cross_entropy_loss",
    
    # Checkpointing
    "CheckpointManager",
    "CheckpointState",
    "CheckpointInfo",
    "save_checkpoint",
    "load_checkpoint",
    "save_model",
    "load_model",
    "save_optimizer",
    "load_optimizer",
    
    # Generation
    "generate",
    "generate_streaming",
    "complete",
    "batch_generate",
    "sample_token",
    
    # Modules
    "AttentionBlock",
    "CausalSelfAttention",
    "KVCache",
    "MambaBlock",
    "MambaState",
    "MLP",
    "RMSNorm",
    "Embedding",
    "RotaryEmbedding",
]
