"""
SentiFormer - Basic Usage Examples
"""

from sentiment_utils import SentimentAnalyzer

# ─────────────────────────────────────────────
# 1. Load the model
# ─────────────────────────────────────────────
analyzer = SentimentAnalyzer()
analyzer.load_model_from_checkpoint(
    model_path="sentiformer.pth",
    tokenizer_path="tokenizer.json"
)

# ─────────────────────────────────────────────
# 2. Single prediction
# ─────────────────────────────────────────────
text = "The food was absolutely fantastic, I'll definitely be back!"
sentiment, confidence, probabilities = analyzer.predict_sentiment(text)

print("─" * 50)
print(f"Input:      {text}")
print(f"Sentiment:  {sentiment}")
print(f"Confidence: {confidence:.2%}")
print()

# ─────────────────────────────────────────────
# 3. Show full probability distribution
# ─────────────────────────────────────────────
print("Class Probabilities:")
for label, prob in probabilities.items():
    bar = "█" * int(prob * 30)
    print(f"  {label:<35} {prob:.2%}  {bar}")

# ─────────────────────────────────────────────
# 4. Batch prediction
# ─────────────────────────────────────────────
reviews = [
    "Worst experience of my life. Never coming back.",
    "It was okay, nothing special.",
    "Great service. Waited only three hours.",   # sarcasm
    "Absolutely loved everything about this place!",
]

print("\n" + "─" * 50)
print("Batch Predictions:\n")
for review in reviews:
    sentiment, confidence, _ = analyzer.predict_sentiment(review)
    print(f"  Review:     {review}")
    print(f"  Prediction: {sentiment} ({confidence:.2%})\n")
