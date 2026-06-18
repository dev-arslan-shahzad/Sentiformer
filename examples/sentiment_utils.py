import torch
import torch.nn as nn
import math
import os
from tokenizers import Tokenizer
from tokenizers.processors import BertProcessing

# Model Architecture Classes
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerInputLayer(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        B, T, D = x.size()
        Q = self.q_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(attn), V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_linear(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_ff//2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(d_ff//2, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.linear3(self.dropout(self.relu2(self.linear2(self.relu1(self.linear1(x))))))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout):
        super().__init__()
        self.input_layer = TransformerInputLayer(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids, mask=None):
        x = self.input_layer(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model, num_heads, d_ff, num_layers, max_len, dropout):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, mask=None):
        x = self.encoder(input_ids, mask)
        cls_out = x[:, 0, :]
        return self.classifier(cls_out)

# Model Configuration
MODEL_CONFIG = {
    'D_MODEL': 128,
    'NUM_HEADS': 4,
    'D_FF': 512,
    'NUM_LAYERS': 2,
    'MAX_LEN': 256,
    'DROPOUT': 0.1,
    'NUM_CLASSES': 5,  # Yelp review 5-star rating
    'VOCAB_SIZE': 30000
}

# Sentiment mapping
SENTIMENT_MAP = {
    0: "⭐ Very Negative (1 star)",
    1: "⭐⭐ Negative (2 stars)", 
    2: "⭐⭐⭐ Neutral (3 stars)",
    3: "⭐⭐⭐⭐ Positive (4 stars)",
    4: "⭐⭐⭐⭐⭐ Very Positive (5 stars)"
}

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        
    # def load_tokenizer(self, tokenizer_path="tokenizer.json"):
    #     """Load the BPE tokenizer from local file"""
    #     if not os.path.exists(tokenizer_path):
    #         raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        
    #     # Initialize tokenizer with same config as training
    #     SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    #     tok = ByteLevelBPETokenizer(lowercase=True, add_prefix_space=True)
        
    #     # Load the trained tokenizer
    #     with open(tokenizer_path, "r", encoding="utf-8") as f:
    #         tokenizer_str = f.read()

        
    #     tok = ByteLevelBPETokenizer.from_str(tokenizer_str)
        
    #     # Set up post-processor and padding/truncation
    #     tok._tokenizer.post_processor = BertProcessing(
    #         ("[SEP]", tok.token_to_id("[SEP]")),
    #         ("[CLS]", tok.token_to_id("[CLS]"))
    #     )
    #     tok.enable_truncation(max_length=MODEL_CONFIG['MAX_LEN'])
    #     tok.enable_padding(
    #         length=MODEL_CONFIG['MAX_LEN'],
    #         pad_id=tok.token_to_id("[PAD]"),
    #         pad_token="[PAD]"
    #     )
        
    #     return tok

    def load_tokenizer(self, tokenizer_path="tokenizer.json"):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        tok = Tokenizer.from_file(tokenizer_path)

        tok.post_processor = BertProcessing(
            ("[SEP]", tok.token_to_id("[SEP]")),
            ("[CLS]", tok.token_to_id("[CLS]"))
        )
        tok.enable_truncation(max_length=MODEL_CONFIG['MAX_LEN'])
        tok.enable_padding(
            length=MODEL_CONFIG['MAX_LEN'],
            pad_id=tok.token_to_id("[PAD]"),
            pad_token="[PAD]"
        )

        return tok


    def load_model_from_checkpoint(self, model_path="transformer_classifier_checkpoint_best_best.pth", tokenizer_path="tokenizer.json"):
        """Load model and tokenizer from local files"""
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        vocab_size = self.tokenizer.get_vocab_size()
        print(f"Tokenizer loaded with vocab size: {vocab_size}")
        
        # Initialize model architecture
        print("Initializing model...")
        self.model = TransformerClassifier(
            vocab_size=vocab_size,
            num_classes=MODEL_CONFIG['NUM_CLASSES'],
            d_model=MODEL_CONFIG['D_MODEL'],
            num_heads=MODEL_CONFIG['NUM_HEADS'],
            d_ff=MODEL_CONFIG['D_FF'],
            num_layers=MODEL_CONFIG['NUM_LAYERS'],
            max_len=MODEL_CONFIG['MAX_LEN'],
            dropout=MODEL_CONFIG['DROPOUT']
        )
        
        # Load model weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print("Loading model weights...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        return self.model, self.tokenizer, self.device

    def predict_sentiment(self, text):
        print("Entered prediction function")
        if self.model is None or self.tokenizer is None:
            return "Error: Model not loaded. Please ensure model files are available.", 0.0, {}

        if not text.strip():
            return "Please enter some text to analyze.", 0.0, {}

        try:
            print("Tokenizing input...")
            enc = self.tokenizer.encode(text)
            input_ids = torch.tensor([enc.ids]).to(self.device)

            print("Running model...")
            with torch.no_grad():
                logits = self.model(input_ids)
                print("Got logits:", logits)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

            print("Post-processing result...")
            sentiment = SENTIMENT_MAP.get(pred, f"Class {pred}")
            prob_dict = {SENTIMENT_MAP[i]: float(probs[0][i].item()) for i in SENTIMENT_MAP}

            print("Returning result...")
            return sentiment, confidence, prob_dict

        except Exception as e:
            print("Prediction error:", str(e))
            return f"Error during prediction: {str(e)}", 0.0, {}


    def is_loaded(self):
        """Check if model and tokenizer are loaded"""
        return self.model is not None and self.tokenizer is not None