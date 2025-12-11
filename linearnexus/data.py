"""Data loading utilities for LLM training.

Provides minimal, nanoGPT-style data pipeline:
- CharTokenizer: Character-level tokenization
- BPETokenizer: Tiktoken-based BPE (optional dependency)
- TextDataset: Memory-mapped text dataset
- DataLoader: Batched sequence generation

Example:
    tokenizer = CharTokenizer.from_file("data/shakespeare.txt")
    dataset = TextDataset("data/shakespeare.txt", tokenizer, seq_len=256)
    loader = DataLoader(dataset, batch_size=4)
    
    for batch in loader:
        tokens = batch["input_ids"]  # [batch, seq_len]
        labels = batch["labels"]     # [batch, seq_len]
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


# -----------------------------------------------------------------------------
# Tokenizer Interface
# -----------------------------------------------------------------------------

class Tokenizer(ABC):
    """Abstract tokenizer interface."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Size of vocabulary."""
        pass
    
    @property
    @abstractmethod
    def eos_token_id(self) -> Optional[int]:
        """End-of-sequence token ID."""
        pass
    
    @property
    @abstractmethod
    def pad_token_id(self) -> Optional[int]:
        """Padding token ID."""
        pass


# -----------------------------------------------------------------------------
# Character Tokenizer (nanoGPT-style)
# -----------------------------------------------------------------------------

class CharTokenizer(Tokenizer):
    """Character-level tokenizer.
    
    Simple and effective for small-scale experiments. Each unique
    character in the training data becomes a token.
    
    Args:
        chars: String of all unique characters in vocabulary.
    """
    
    def __init__(self, chars: str):
        self.chars = sorted(set(chars))
        self._stoi = {ch: i for i, ch in enumerate(self.chars)}
        self._itos = {i: ch for i, ch in enumerate(self.chars)}
        self._vocab_size = len(self.chars)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "CharTokenizer":
        """Create tokenizer from text file."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(text)
    
    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Create tokenizer from text string."""
        return cls(text)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to character IDs."""
        return [self._stoi.get(ch, 0) for ch in text]
    
    def decode(self, ids: List[int]) -> str:
        """Decode character IDs to text."""
        return "".join(self._itos.get(i, "") for i in ids)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def eos_token_id(self) -> Optional[int]:
        return None  # No special EOS token
    
    @property
    def pad_token_id(self) -> Optional[int]:
        return None  # No padding needed for char-level
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file."""
        with open(path, "wb") as f:
            pickle.dump({"chars": self.chars}, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharTokenizer":
        """Load tokenizer from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["chars"])


# -----------------------------------------------------------------------------
# BPE Tokenizer (tiktoken wrapper)
# -----------------------------------------------------------------------------

class BPETokenizer(Tokenizer):
    """BPE tokenizer using tiktoken.
    
    Wraps tiktoken encodings for GPT-2, GPT-4, etc.
    Requires: pip install tiktoken
    
    Args:
        encoding_name: Tiktoken encoding name ("gpt2", "cl100k_base", etc.)
    """
    
    def __init__(self, encoding_name: str = "gpt2"):
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for BPETokenizer. "
                "Install with: pip install tiktoken"
            )
        
        self._enc = tiktoken.get_encoding(encoding_name)
        self._encoding_name = encoding_name
    
    def encode(self, text: str) -> List[int]:
        """Encode text to BPE token IDs."""
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})
    
    def decode(self, ids: List[int]) -> str:
        """Decode BPE token IDs to text."""
        return self._enc.decode(ids)
    
    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab
    
    @property
    def eos_token_id(self) -> Optional[int]:
        return self._enc.eot_token
    
    @property
    def pad_token_id(self) -> Optional[int]:
        return self._enc.eot_token  # Use EOT as padding


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class TextDataset:
    """Memory-mapped text dataset for efficient loading.
    
    Loads tokenized data into memory-mapped numpy array for efficient
    random access without loading entire file into RAM.
    
    Args:
        path: Path to text file or pre-tokenized .bin file.
        tokenizer: Tokenizer for encoding text.
        seq_len: Sequence length for each sample.
        cache_dir: Directory for cached tokenized data.
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        tokenizer: Tokenizer,
        seq_len: int = 256,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Determine cache path
        if cache_dir is None:
            cache_dir = self.path.parent / ".cache"
        self.cache_dir = Path(cache_dir)
        
        # Load or create tokenized data
        self._data = self._load_or_tokenize()
        
        # Calculate number of samples
        # Each sample needs seq_len + 1 tokens (input + target)
        self._num_samples = max(0, len(self._data) - seq_len)
    
    def _load_or_tokenize(self) -> np.ndarray:
        """Load from cache or tokenize text file."""
        # Check for .bin file (pre-tokenized)
        if self.path.suffix == ".bin":
            return np.memmap(self.path, dtype=np.int32, mode="r")
        
        # Check cache
        cache_path = self.cache_dir / f"{self.path.stem}.bin"
        if cache_path.exists():
            return np.memmap(cache_path, dtype=np.int32, mode="r")
        
        # Tokenize and cache
        print(f"Tokenizing {self.path}...")
        with open(self.path, "r", encoding="utf-8") as f:
            text = f.read()
        
        tokens = self.tokenizer.encode(text)
        data = np.array(tokens, dtype=np.int32)
        
        # Save to cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fp = np.memmap(cache_path, dtype=np.int32, mode="w+", shape=data.shape)
        fp[:] = data
        fp.flush()
        print(f"Cached {len(data)} tokens to {cache_path}")
        
        return np.memmap(cache_path, dtype=np.int32, mode="r")
    
    def __len__(self) -> int:
        return self._num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample.
        
        Returns:
            Dict with "input_ids" and "labels" arrays.
        """
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        # Get seq_len + 1 tokens
        chunk = self._data[idx : idx + self.seq_len + 1]
        
        return {
            "input_ids": chunk[:-1].copy(),
            "labels": chunk[1:].copy(),
        }
    
    def get_batch(
        self,
        indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Get a batch of samples.
        
        Args:
            indices: Array of sample indices.
            
        Returns:
            Dict with batched "input_ids" and "labels".
        """
        batch_size = len(indices)
        input_ids = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        labels = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        
        for i, idx in enumerate(indices):
            sample = self[idx]
            input_ids[i] = sample["input_ids"]
            labels[i] = sample["labels"]
        
        return {"input_ids": input_ids, "labels": labels}


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------

class DataLoader:
    """Simple batched data loader with shuffling.
    
    Args:
        dataset: TextDataset to load from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle at each epoch.
        drop_last: Whether to drop incomplete final batch.
        seed: Random seed for shuffling.
    """
    
    def __init__(
        self,
        dataset: TextDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        
        self._indices = np.arange(len(dataset))
    
    def __len__(self) -> int:
        n = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            n += 1
        return n
    
    def __iter__(self) -> Iterator[Dict[str, Array]]:
        """Iterate over batches."""
        indices = self._indices.copy()
        
        if self.shuffle:
            self.rng.shuffle(indices)
        
        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            
            if end > len(indices):
                if self.drop_last:
                    break
                end = len(indices)
            
            batch_indices = indices[start:end]
            batch = self.dataset.get_batch(batch_indices)
            
            # Convert to JAX arrays
            yield {
                "input_ids": jnp.array(batch["input_ids"]),
                "labels": jnp.array(batch["labels"]),
            }
    
    def sample_batch(self) -> Dict[str, Array]:
        """Sample a random batch (useful for evaluation)."""
        indices = self.rng.choice(len(self.dataset), size=self.batch_size, replace=False)
        batch = self.dataset.get_batch(indices)
        return {
            "input_ids": jnp.array(batch["input_ids"]),
            "labels": jnp.array(batch["labels"]),
        }


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def prepare_dataset(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    tokenizer: Tokenizer,
) -> int:
    """Tokenize a text file and save as binary.
    
    Args:
        input_path: Path to input text file.
        output_path: Path for output .bin file.
        tokenizer: Tokenizer to use.
        
    Returns:
        Number of tokens.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Tokenizing {len(text)} characters...")
    tokens = tokenizer.encode(text)
    data = np.array(tokens, dtype=np.int32)
    
    print(f"Writing {len(data)} tokens to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(output_path)
    
    return len(data)


def download_shakespeare(data_dir: Union[str, Path] = "data") -> Path:
    """Download tiny Shakespeare dataset.
    
    Returns:
        Path to downloaded file.
    """
    import urllib.request
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_path = data_dir / "shakespeare.txt"
    
    if not output_path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved to {output_path}")
    else:
        print(f"Using cached {output_path}")
    
    return output_path
