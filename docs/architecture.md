# SentiFormer — Architecture Deep Dive

## Overview

SentiFormer is a Transformer encoder trained from scratch for 5-class sentiment classification. It does not use any pre-trained weights. Every component — the tokenizer, the embeddings, and the attention layers — is learned entirely from the training corpus.

---

## Tokenizer

**Type:** Byte-Level Byte Pair Encoding (BPE)  
**Library:** HuggingFace `tokenizers`  
**Vocabulary size:** 30,000  
**Minimum frequency:** 2  
**Max sequence length:** 256  
**Lowercase:** True

### Special Tokens

| Token | Purpose |
|---|---|
| `[PAD]` | Padding to fixed length |
| `[UNK]` | Unknown/out-of-vocab subwords |
| `[CLS]` | Sequence classification token (position 0) |
| `[SEP]` | End-of-sequence marker |

### Why BPE?

Word-level tokenizers fail silently on misspellings, slang, emojis, and rare words — all common in review text. BPE decomposes unknown words into known subword units:

```
"unbelievably" → ["un", "believ", "ably"]
"gr8"          → ["gr", "8"]
```

This gives the model a fighting chance on out-of-vocabulary input without requiring a massive vocabulary.

---

## Embedding Layer

```python
nn.Embedding(vocab_size, d_model=128)
```

Converts integer token IDs into 128-dimensional dense vectors. These are not frozen — the embedding weights are learned jointly with the rest of the model.

Output shape: `(batch, seq_len, 128)`

---

## Positional Encoding

Transformers process all tokens in parallel and have no built-in sense of order. Sinusoidal positional encodings are added to the embeddings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

This injects sequence position information without adding trainable parameters. The sinusoidal pattern also generalizes to sequence lengths not seen during training.

---

## Transformer Encoder Block

Two identical blocks are stacked. Each block contains:

### 1. Multi-Head Self-Attention

4 attention heads, each operating in 32-dimensional subspace (128 / 4 = 32).

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```

Each head learns to attend to different relationships:
- Negation scope (e.g., "not good")
- Adjective–noun binding
- Long-range sentiment dependencies

Outputs of all heads are concatenated and projected back to 128 dimensions.

### 2. Residual Connection + LayerNorm

```
x = LayerNorm(x + Attention(x))
```

Residual connections preserve gradient flow across layers. LayerNorm stabilizes activations per sample.

### 3. Feed-Forward Network

```
128 → Linear → ReLU → 512 → Linear → ReLU → 256 → Linear → 128
```

Applied position-wise. Increases representational capacity between attention layers.

### 4. Residual Connection + LayerNorm (again)

```
x = LayerNorm(x + FFN(x))
```

---

## Classification Head

After two encoder blocks, the `[CLS]` token representation at position 0 is used as the sequence embedding:

```python
cls_out = encoder_output[:, 0, :]  # shape: (batch, 128)
logits  = nn.Linear(128, 5)(cls_out)
```

This follows the standard BERT-style classification approach. The `[CLS]` token attends to all other tokens across both encoder layers, accumulating a global representation of the input.

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | Adam (lr=3e-4, defaults betas) |
| Loss | CrossEntropyLoss |
| Epochs | 30 |
| Batch size | 64 |
| Dropout | 0.1 (attention + FFN + embedding) |
| Checkpointing | Best validation loss |

---

## Parameter Count (approximate)

| Component | Parameters |
|---|---|
| Token embedding | 30,000 × 128 = 3.84M |
| Positional encoding | 0 (fixed) |
| Encoder block × 2 | ~0.66M |
| Classifier head | 128 × 5 = 640 |
| **Total** | **~4.5M** |

---

## Inference

```python
# Tokenize
enc = tokenizer.encode(text)            # BPE encode + pad/truncate to 256
input_ids = torch.tensor([enc.ids])     # shape: (1, 256)

# Forward pass
logits = model(input_ids)               # shape: (1, 5)
probs  = torch.softmax(logits, dim=1)   # shape: (1, 5)
pred   = torch.argmax(probs, dim=1)     # 0–4
```

No gradient computation is performed during inference (`torch.no_grad()`).
