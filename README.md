# SentiFormer

**A custom Transformer-based sentiment analysis model trained from scratch on Yelp reviews.**

SentiFormer implements a full Transformer encoder architecture in PyTorch — no pre-trained BERT or RoBERTa — paired with a custom Byte-Level BPE tokenizer trained on review corpora. It classifies text into five sentiment categories (1–5 stars) with particular strength on strongly positive and strongly negative reviews.

---

## Authors

- **Muhammad Arslan Shahzad**
- **Zaid Khan**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Model Files](#model-files)
- [Dataset](#dataset)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Citation](#citation)

---

## Overview

| Property | Value |
|---|---|
| Task | 5-class sentiment classification |
| Architecture | Custom Transformer Encoder |
| Tokenizer | Byte-Level BPE (trained from scratch) |
| Vocab Size | 30,000 |
| Parameters | ~3M |
| Test Accuracy | 63% (balanced 50k samples) |
| Framework | PyTorch |

### Classes

| Label | Sentiment |
|---|---|
| 0 | ⭐ Very Negative (1 star) |
| 1 | ⭐⭐ Negative (2 stars) |
| 2 | ⭐⭐⭐ Neutral (3 stars) |
| 3 | ⭐⭐⭐⭐ Positive (4 stars) |
| 4 | ⭐⭐⭐⭐⭐ Very Positive (5 stars) |

---

## Architecture

```
Input Text
    ↓
BPE Tokenizer (vocab=30k, max_len=256)
    ↓
Token Embedding (d_model=128)
    ↓
Sinusoidal Positional Encoding
    ↓
Transformer Encoder Block × 2
  ├─ Multi-Head Self-Attention (4 heads)
  ├─ Add & LayerNorm
  ├─ Feed-Forward (128 → 512 → 256 → 128)
  └─ Add & LayerNorm
    ↓
[CLS] Token Representation
    ↓
Linear Classifier (128 → 5)
    ↓
Star Rating Prediction
```

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| d_model | 128 |
| Attention heads | 4 |
| Feed-forward dim | 512 |
| Encoder layers | 2 |
| Max sequence length | 256 |
| Dropout | 0.1 |
| Batch size | 64 |
| Learning rate | 3e-4 |
| Epochs | 30 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

---

## Performance

### Overall Accuracy: **63%** on 50,000 balanced test reviews

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 1 Star | 0.72 | 0.79 | 0.76 |
| 2 Star | 0.57 | 0.55 | 0.56 |
| 3 Stars | 0.59 | 0.50 | 0.54 |
| 4 Stars | 0.54 | 0.53 | 0.54 |
| 5 Stars | 0.69 | 0.76 | 0.72 |
| **Macro Avg** | **0.62** | **0.63** | **0.62** |

The model achieves highest recall on extreme sentiments (1-star and 5-star), consistent with findings in sentiment analysis literature. Mixed-sentiment reviews (2–4 stars) are harder to classify due to overlapping linguistic signals.

---

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- tokenizers ≥ 0.13

```bash
pip install torch tokenizers
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
from sentiment_utils import SentimentAnalyzer

# Initialize and load model
analyzer = SentimentAnalyzer()
analyzer.load_model_from_checkpoint(
    model_path="sentiformer.pth",
    tokenizer_path="tokenizer.json"
)

# Predict sentiment
text = "The food was absolutely fantastic, I'll definitely be back!"
sentiment, confidence, probabilities = analyzer.predict_sentiment(text)

print(f"Sentiment:  {sentiment}")
print(f"Confidence: {confidence:.2%}")
print("\nClass Probabilities:")
for label, prob in probabilities.items():
    print(f"  {label}: {prob:.2%}")
```

**Output:**
```
Sentiment:  ⭐⭐⭐⭐⭐ Very Positive (5 stars)
Confidence: 82.14%

Class Probabilities:
  ⭐ Very Negative (1 star): 1.23%
  ⭐⭐ Negative (2 stars): 2.11%
  ⭐⭐⭐ Neutral (3 stars): 5.88%
  ⭐⭐⭐⭐ Positive (4 stars): 8.64%
  ⭐⭐⭐⭐⭐ Very Positive (5 stars): 82.14%
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for more examples.

---

## API Reference

### `SentimentAnalyzer`

The main class for loading and running inference.

#### `load_model_from_checkpoint(model_path, tokenizer_path)`

Loads model weights and tokenizer from disk.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | `"sentiformer.pth"` | Path to `.pth` checkpoint file |
| `tokenizer_path` | `str` | `"tokenizer.json"` | Path to BPE tokenizer JSON |

Automatically detects and uses CUDA if available.

#### `predict_sentiment(text)`

Runs inference on a single text string.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Review text to classify |

**Returns:** `(sentiment: str, confidence: float, probabilities: dict)`

| Return | Type | Description |
|---|---|---|
| `sentiment` | `str` | Human-readable label with star emojis |
| `confidence` | `float` | Probability of predicted class (0–1) |
| `probabilities` | `dict` | Full probability distribution across all 5 classes |

#### `is_loaded()`

Returns `True` if model and tokenizer are both loaded.

---

## Model Files

Download the model checkpoint and tokenizer and place them in your project directory:

| File | Description |
|---|---|
| `sentiformer.pth` | Model weights (~12 MB) |
| `tokenizer.json` | BPE tokenizer vocabulary and merges |

> **Note:** Model and tokenizer files are not included in this repository due to size. See [Releases](../../releases) or contact the authors.

---

## Dataset

### Yelp Review Full

- Large-scale real-world review corpus
- Balanced 1–5 star distribution
- Used for primary training and evaluation

### Sarcasm Dataset (Custom)

A supplementary dataset of sarcastic reviews was integrated to improve robustness against sentiment reversal — cases where positive words are used with negative intent:

> *"Great service. Waited only three hours."*

Standard sentiment models often misclassify such examples. Training on sarcastic examples improves the model's contextual understanding.

---

## Limitations

- **Moderate accuracy (63%):** Custom transformers trained from scratch are competitive but below large pre-trained LLMs fine-tuned on similar tasks.
- **Middle-class confusion:** 2–4 star reviews share overlapping language and are the primary source of errors.
- **Max 256 tokens:** Long reviews are truncated; content beyond this length is ignored.
- **Small capacity:** 2 encoder layers and 128-dim embeddings limit representational power.

---

## Future Work

- Increase encoder depth (2 → 6 layers) and embedding dim (128 → 512)
- Add padding-aware attention masking
- Learning rate scheduling (warmup + cosine decay)
- Early stopping based on validation F1
- Focal loss to address class imbalance
- Expand sarcasm training data
- BERT/RoBERTa baseline comparison

---

## Repository Structure

```
SentiFormer/     # Model architecture, config, and SentimentAnalyzer class
├── examples/
│   └── basic_usage.py      # Usage examples
│   └── sentiment_utils.py      # Usage examples
├── docs/
│   └── architecture.md     # Detailed architecture documentation
├── requirements.txt
├── LICENSE
└── sentiformer.pth
└── tokenizer.json
└── Sentiment_Analysis.ipynb
├── Integration/
│   └── Realtime Integration.pdf 
```

---

## Citation

If you use SentiFormer in your research or projects, please cite:

```bibtex
@misc{sentiformer2025,
  title     = {SentiFormer: Custom Transformer-Based Sentiment Analysis},
  author    = {Muhammad Arslan Shahzad and Zaid Khan},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/dev-arslan-shahzad/Sentiformer}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
