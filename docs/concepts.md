# Concepts Guide

**Understanding the fundamentals for beginners**

This guide explains the core concepts behind language models and LinearNexus. No prior ML knowledge required.

---

## Table of Contents

1. [What is a Language Model?](#what-is-a-language-model)
2. [How Do LLMs Generate Text?](#how-do-llms-generate-text)
3. [What is a Token?](#what-is-a-token)
4. [The Transformer Architecture](#the-transformer-architecture)
5. [What is Mamba?](#what-is-mamba)
6. [Training vs Inference](#training-vs-inference)
7. [Key Hyperparameters](#key-hyperparameters)
8. [Glossary](#glossary)

---

## What is a Language Model?

A **language model (LM)** is a program that predicts the next word (or character) given previous words.

```
Input:  "The cat sat on the"
Output: "mat" (predicted next word)
```

Large Language Models (LLMs) like GPT-4, Claude, and Llama are just very large language models trained on massive amounts of text.

**How it works**:
1. Take some text as input
2. Convert it to numbers (tokens)
3. Pass through a neural network
4. Get probabilities for each possible next token
5. Sample or pick the most likely one

---

## How Do LLMs Generate Text?

LLMs generate text **one token at a time** in a loop:

```
Step 1: "Once upon a" → predict "time" 
Step 2: "Once upon a time" → predict "there"
Step 3: "Once upon a time there" → predict "was"
...and so on
```

This is called **autoregressive generation** — each prediction becomes part of the input for the next prediction.

### Sampling Strategies

When picking the next token, we have options:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Greedy** | Always pick highest probability | Deterministic, may be boring |
| **Temperature** | Scale probabilities (higher = more random) | Control creativity |
| **Top-k** | Only consider top k tokens | Reduce nonsense |
| **Top-p** | Only consider tokens until cumulative prob > p | Adaptive filtering |

```python
# Greedy (temperature = 0)
next_token = argmax(probabilities)

# Temperature sampling
scaled_probs = softmax(logits / temperature)
next_token = sample(scaled_probs)
```

---

## What is a Token?

A **token** is the basic unit the model works with. It could be:

| Type | Example | Vocabulary Size |
|------|---------|-----------------|
| **Character** | `h`, `e`, `l`, `l`, `o` | ~100 |
| **Subword (BPE)** | `hello`, ` world` | ~50,000 |
| **Word** | `hello`, `world` | ~100,000+ |

Most modern LLMs use **subword tokenization** (like BPE - Byte Pair Encoding).

```python
# Character tokenizer (simple)
"hello" → [7, 4, 11, 11, 14]

# BPE tokenizer (GPT-2 style)
"Hello, world!" → [15496, 11, 995, 0]
```

**Why subwords?**
- Smaller vocabulary than words
- Can handle any text (no "unknown" tokens)
- Balances efficiency and flexibility

---

## The Transformer Architecture

**Transformers** are the architecture behind GPT, BERT, and most modern LLMs.

### Key Components

```
┌──────────────────────────────────────┐
│           Transformer Block           │
├──────────────────────────────────────┤
│  ┌─────────────────────────────────┐ │
│  │        Self-Attention           │ │  ← "Look at other tokens"
│  └─────────────────────────────────┘ │
│                  ↓                    │
│  ┌─────────────────────────────────┐ │
│  │         Feed-Forward            │ │  ← "Think about what I saw"
│  │           (MLP)                 │ │
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
                   ↓
           (Repeat N times)
```

### Self-Attention

**Attention** lets each token "look at" all other tokens to understand context:

```
"The cat sat on the mat because it was tired"
                                   ↑
                            What does "it" refer to?
                            Attention helps figure out
                            it refers to "cat"
```

**How it works** (simplified):
1. Each token creates a Query ("What am I looking for?")
2. Each token creates a Key ("What do I contain?")
3. Each token creates a Value ("What information do I have?")
4. Queries match with Keys to find relevant tokens
5. Values from relevant tokens are combined

### Causal Masking

For language generation, tokens can only look at **previous** tokens (not future ones):

```
Token 1: can see [1]
Token 2: can see [1, 2]
Token 3: can see [1, 2, 3]
Token 4: can see [1, 2, 3, 4]
```

This prevents "cheating" during training.

### The Problem with Attention

Attention's complexity is **O(n²)** — it scales quadratically with sequence length.

```
Sequence length:  128   512   2048   8192
Memory:           1x    16x   256x   4096x
```

This is why long documents are challenging for transformers.

---

## What is Mamba?

**Mamba** is an alternative to attention using **State-Space Models (SSMs)**.

### Key Idea

Instead of looking at all tokens at once (attention), Mamba:
1. Maintains a **hidden state** that summarizes history
2. Updates this state as each token arrives
3. Complexity is **O(n)** — linear with sequence length!

```
Attention:  Every token talks to every other token
Mamba:      Each token updates a running summary
```

### Visual Comparison

**Attention**:
```
Token1 ←→ Token2 ←→ Token3 ←→ Token4
   ↕         ↕         ↕         ↕
(All tokens connect to all others)
```

**Mamba**:
```
Token1 → [State] → Token2 → [State] → Token3 → [State] → Token4
                    ↓                   ↓                   ↓
              (State carries          (State updated)   (Final output)
               information forward)
```

### When to Use What?

| Use Case | Attention | Mamba |
|----------|-----------|-------|
| Short sequences (<2K) | ✓ Great | ✓ Good |
| Long sequences (>8K) | ✗ Slow | ✓ Great |
| In-context learning | ✓ Great | ? Developing |
| Fast inference | ✗ Needs cache | ✓ Natural |

### Hybrid (Jamba)

**Jamba** combines both — mostly Mamba layers with occasional attention:

```
Layer 1:  Mamba
Layer 2:  Mamba
Layer 3:  Mamba
Layer 4:  Attention  ← Global context
Layer 5:  Mamba
Layer 6:  Mamba
...
```

This gets benefits of both: fast processing + occasional global context.

---

## Training vs Inference

### Training

**Goal**: Adjust model weights to predict text better.

```
Data: "The cat sat on the mat"

Input:  [The, cat, sat, on, the]
Target: [cat, sat, on, the, mat]

Model predicts each next token, compares to target,
calculates error (loss), updates weights.
```

**Key concepts**:
- **Loss**: How wrong the predictions are (lower = better)
- **Gradient**: Direction to adjust weights
- **Learning rate**: How big the adjustments are
- **Epoch**: One pass through all training data
- **Batch**: Group of examples processed together

### Inference

**Goal**: Generate text using trained weights.

```
Prompt: "Once upon a"
Output: "Once upon a time there was a princess..."
```

The model predicts one token, appends it, predicts the next, and so on.

**Key concepts**:
- **Temperature**: Controls randomness (0 = deterministic, 1+ = creative)
- **Top-k/Top-p**: Limits which tokens to consider
- **KV Cache**: Stores computed values to avoid re-computation

---

## Key Hyperparameters

### Model Size

| Parameter | What it controls | Typical values |
|-----------|------------------|----------------|
| `hidden_size` | Width of model | 256, 512, 768, 1024 |
| `n_layer` | Depth (number of blocks) | 4, 6, 12, 24 |
| `n_head` | Attention heads | 4, 8, 12, 16 |
| `vocab_size` | Number of tokens | 256 (char), 50257 (BPE) |

**Rough parameter count**: `12 × n_layer × hidden_size²`

```
GPT-2 Small:  12 layers, 768 hidden → ~124M parameters
GPT-2 Medium: 24 layers, 1024 hidden → ~350M parameters
```

### Training

| Parameter | What it controls | Typical values |
|-----------|------------------|----------------|
| `learning_rate` | Step size for updates | 1e-4 to 1e-3 |
| `batch_size` | Examples per update | 4, 8, 16, 32 |
| `max_steps` | Total training steps | 1000 - 100000 |
| `warmup_steps` | LR ramp-up period | 100 - 1000 |
| `weight_decay` | Regularization | 0.01 - 0.1 |

### Generation

| Parameter | What it controls | Typical values |
|-----------|------------------|----------------|
| `temperature` | Randomness | 0.7 - 1.0 |
| `top_k` | Consider top k tokens | 40, 50, 100 |
| `top_p` | Nucleus sampling | 0.9, 0.95 |
| `max_tokens` | Generation length | 100 - 2000 |

---

## Glossary

### A-E

**Attention**: Mechanism for tokens to look at each other.

**Autoregressive**: Generating one token at a time, using previous tokens as input.

**Batch**: Group of training examples processed together.

**BPE (Byte Pair Encoding)**: Subword tokenization algorithm.

**Causal**: Only looking at past tokens, not future ones.

**Embedding**: Converting tokens to vectors.

**Epoch**: One complete pass through training data.

### F-L

**Fine-tuning**: Training a pre-trained model on new data.

**Gradient**: Direction to update weights to reduce loss.

**Hidden size**: Dimension of internal representations.

**Inference**: Using a trained model to generate text.

**KV Cache**: Stored key/value tensors for fast generation.

**Layer**: One transformer/Mamba block.

**Loss**: Measure of prediction error.

### M-R

**Mamba**: State-space model alternative to attention.

**MLP**: Multi-layer perceptron (feed-forward network).

**Parameter**: A learnable weight in the model.

**Pre-training**: Initial training on large text corpus.

**Prompt**: Input text to start generation.

**RMSNorm**: Layer normalization variant.

### S-Z

**Sampling**: Choosing next token from probability distribution.

**Sequence length**: Number of tokens in input.

**SFT (Supervised Fine-Tuning)**: Training on input-output pairs.

**SSM (State-Space Model)**: Recurrent model like Mamba.

**Temperature**: Scaling factor for randomness in sampling.

**Token**: Basic unit of text for the model.

**Transformer**: Architecture using self-attention.

**Vocabulary**: Set of all possible tokens.

**Weight**: Learnable parameter in neural network.

---

## Further Reading

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual explanation of attention
- [Mamba Paper](https://arxiv.org/abs/2312.00752) — Original Mamba research
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Inspiration for LinearNexus
- [Andrej Karpathy's YouTube](https://www.youtube.com/@AndrejKarpathy) — Excellent tutorials

---

## Next Steps

Ready to code? Head to [GETTING_STARTED.md](../GETTING_STARTED.md)!
