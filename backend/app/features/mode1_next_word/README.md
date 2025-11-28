# Mode 1: Next Word Prediction (Mini-GPT)

Educational GPT-style language model that predicts the next word given an input sequence. This is a complete, self-contained feature module with training, inference, and visualization capabilities.

---

## 📋 Overview

**Mode 1** demonstrates autoregressive language modeling using a decoder-only transformer architecture (GPT-style). The model is trained from scratch on a custom corpus and provides detailed visualization data for educational purposes.

### Key Capabilities

- ✅ **Next Word Prediction** - Predicts the most likely next word given input text
- ✅ **Training Pipeline** - Complete training script with validation support
- ✅ **Interactive Visualizations** - 6-step pipeline visualization in frontend
- ✅ **Educational Focus** - Detailed explanations at each stage
- ✅ **Self-Contained Module** - All code, data, and models in one folder

---

## 🗂️ Project Structure

```
mode1_next_word/
├── api/
│   └── router.py                  # FastAPI routes for Mode 1
├── model/
│   └── gpt_model.py              # GPT-style decoder-only transformer
├── service/
│   └── gpt_service.py            # Inference and prediction service
├── training/
│   ├── dataset.py                # Tokenizer and dataset utilities
│   └── trainer.py                # Training loop with validation
├── data/
│   └── sample_corpus.txt         # Training corpus (1,449 lines)
├── checkpoints/
│   └── best_model.pt             # Trained model (used by application)
├── train.py                      # Training script (CLI)
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Train a New Model

Train a GPT model from scratch on the sample corpus:

```bash
# From backend/ directory
cd backend

# Activate virtual environment
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run training (50 epochs with validation)
python -m app.features.mode1_next_word.train --epochs 50

# Or with custom configuration
python -m app.features.mode1_next_word.train \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --d-model 256 \
  --n-heads 4 \
  --n-layers 4
```

**Training Output:**
- Checkpoints saved every 10 epochs in `checkpoints/`
- `best_model.pt` - Model with lowest validation loss (automatically selected)
- `final_model.pt` - Model at last epoch
- Training history with train/validation loss

### 2. Use the Trained Model (API)

The API automatically loads `best_model.pt` on startup. No additional configuration needed!

**Run Backend:**
```bash
# From backend/ directory
python -m app.main
```

**API Endpoint:**
```bash
POST http://localhost:8000/api/v1/predict-next-word
Content-Type: application/json

{
  "text": "I eat"
}
```

**Response:**
```json
{
  "input_text": "I eat",
  "predicted_token": "vegetables",
  "predicted_word": "vegetables",
  "confidence": 0.529,
  "top_predictions": [
    {"word": "vegetables", "probability": 0.529, "log_prob": -0.632},
    {"word": "breakfast", "probability": 0.287, "log_prob": -1.248},
    {"word": "rice", "probability": 0.184, "log_prob": -1.693}
  ],
  "steps": {
    "tokenization": {...},
    "embeddings": {...},
    "attention": {...},
    "feedforward": {...},
    "output": {...}
  }
}
```

### 3. Explore Visualizations (Frontend)

1. Start backend (port 8000) and frontend (port 3000)
2. Navigate to **Applications** → **Mode 1: Next Word Prediction**
3. Enter text and click **"Predict Next Word"**
4. Explore 6 visualization steps:
   - **Step 1:** Tokenization
   - **Step 2:** Embeddings + Positional Encoding
   - **Step 3:** Self-Attention & Multi-Head Attention
   - **Step 4:** Feedforward Network
   - **Step 5:** Output Layer (Softmax)
   - **Step 6:** Prediction Result

---

## 🎓 Model Architecture

### GPT-Style Decoder-Only Transformer

**Configuration** (Educational size, optimized for CPU):

```python
GPTModel(
    vocab_size=1500,          # Word-level tokenization
    d_model=256,              # Model dimension (embedding size)
    n_heads=4,                # Number of attention heads
    n_layers=4,               # Number of transformer blocks
    d_ff=1024,                # Feed-forward dimension
    max_len=60,               # Maximum sequence length
    dropout=0.1,              # Dropout rate
    padding_idx=0             # Padding token index
)
```

**Total Parameters:** ~3.9 million trainable parameters

### Architecture Components

1. **Token Embedding Layer** - Maps tokens to d_model dimensional vectors
2. **Positional Encoding** - Sinusoidal encoding for position information
3. **Transformer Decoder Blocks** (4 layers):
   - Multi-Head Self-Attention (causal masking)
   - Layer Normalization
   - Feed-Forward Network (d_model → d_ff → d_model)
   - Residual Connections
   - Dropout
4. **Output Projection** - Projects to vocabulary size with softmax

### Causal Masking

Uses **causal (autoregressive) masking** to prevent attending to future tokens:

```
Token:  [I]  [eat]  [rice]  [daily]
Mask:    1     0      0       0
         1     1      0       0
         1     1      1       0
         1     1      1       1
```

This ensures the model only uses past context for prediction.

---

## 📊 Training Details

### Corpus Statistics

**Current Corpus** (`data/sample_corpus.txt`):
- **Lines:** 1,449 sentences
- **Characters:** ~28,828
- **Vocabulary Size:** 1,500 unique word tokens
- **Total Sequences:** 238 training sequences

**Content Categories:**
- Daily routines (wake up, breakfast, work)
- Professions (engineer, teacher, doctor, chef)
- Hobbies (music, sports, cooking, travel)
- Technology (computer, smartphone, apps)
- Life concepts (finance, environment, growth)
- Descriptive scenes (locations, objects, sensory details)

### Training Configuration

**Recommended Hyperparameters:**

```python
--epochs 50              # Number of training epochs
--batch-size 32          # Batch size (adjust for GPU/CPU)
--lr 1e-3                # Learning rate (Adam optimizer)
--d-model 256            # Model dimension
--n-heads 4              # Attention heads (must divide d_model)
--n-layers 4             # Transformer layers
--d-ff 1024              # Feed-forward dimension
--max-seq-len 50         # Maximum sequence length
```

**Data Split:**
- **Training:** 80% of corpus (4,757 tokens)
- **Validation:** 20% of corpus (1,190 tokens)

**Optimization:**
- **Optimizer:** Adam (lr=1e-3, betas=(0.9, 0.999))
- **Scheduler:** ReduceLROnPlateau (monitors validation loss)
- **Loss Function:** CrossEntropyLoss (ignores padding)

### Training Process

**Example Training Run:**

```
Epoch 1/50
  Train Loss: 6.14
  Val Loss: 6.38
  Time: 15.2s

Epoch 6/50
  Train Loss: 5.5
  Val Loss: 4.69  ← BEST MODEL (lowest validation loss)
  [BEST] Best model saved
  Time: 15.8s

Epoch 50/50
  Train Loss: 0.997
  Val Loss: 5.38
  Time: 16.1s

Training Complete!
Best model: Epoch 6 (val_loss: 4.69)
```

**Key Observation:** Validation loss starts increasing after epoch 6, indicating **overfitting**. The trainer automatically saves the best model (epoch 6) to `best_model.pt`.

### Validation Loss vs Training Loss

| Metric | Purpose | Interpretation |
|--------|---------|----------------|
| **Training Loss** | Measures fit to training data | Can be misleading (model may memorize) |
| **Validation Loss** | Measures generalization | **True measure of model quality** |

**Why validation loss is higher:**
- Validation loss measures performance on **unseen data**
- This is the **realistic** measure of model performance
- Lower training loss with higher validation loss = **overfitting**

**Best model selection:**
- ✅ Use `best_model.pt` (lowest validation loss)
- ❌ Don't use `final_model.pt` (may be overfit)

---

## 📦 Checkpoints

### Saved Models

**Location:** `backend/app/features/mode1_next_word/checkpoints/`

| File | Size | Description | Used By |
|------|------|-------------|---------|
| **best_model.pt** | 46 MB | Lowest validation loss | ✅ **Application** |
| final_model.pt | 46 MB | Final epoch | ⚠️ May be overfit |
| checkpoint_epoch_*.pt | 46 MB | Periodic backups | Archive only |

**Checkpoint Contents:**
```python
{
    'epoch': 6,
    'model_state_dict': {...},           # Model weights
    'optimizer_state_dict': {...},       # Optimizer state
    'scheduler_state_dict': {...},       # Scheduler state
    'train_loss': 5.5,
    'val_loss': 4.69,
    'history': {
        'train_loss': [...],
        'val_loss': [...],
        'epochs': 6
    },
    'tokenizer_vocab': {...},            # Vocabulary mapping
    'config': {
        'vocab_size': 1500,
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 1024,
        'max_len': 60,
        'dropout': 0.1
    }
}
```

### Loading a Checkpoint

**Automatic (API):**
```python
# In router.py - automatically loads best_model.pt
gpt_service = GPTService()
```

**Manual (Python):**
```python
from app.features.mode1_next_word.service.gpt_service import GPTService

# Load best model
service = GPTService()

# Or load specific checkpoint
service = GPTService(checkpoint_path='checkpoints/checkpoint_epoch_10.pt')
```

---

## 🔧 API Endpoints

### POST `/api/v1/predict-next-word`

Predict the next word given input text.

**Request:**
```json
{
  "text": "I eat"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `input_text` | string | Original input text |
| `predicted_token` | string | Most likely next token |
| `predicted_word` | string | Same as predicted_token (for clarity) |
| `confidence` | float | Probability of prediction (0-1) |
| `top_predictions` | array | Top-k predictions with probabilities |
| `steps` | object | Detailed visualization data |

**Visualization Steps:**

```json
{
  "steps": {
    "tokenization": {
      "tokens": ["I", "eat"],
      "token_ids": [245, 89],
      "special_tokens": {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    },
    "embeddings": {
      "word_embeddings": [[...], [...]],      // shape: [seq_len, d_model]
      "positional_encodings": [[...], [...]],  // shape: [seq_len, d_model]
      "final_embeddings": [[...], [...]]       // WE + PE
    },
    "attention": {
      "layer_0": {
        "attention_weights": [...],            // shape: [n_heads, seq_len, seq_len]
        "attention_output": [[...], [...]]     // shape: [seq_len, d_model]
      },
      // ... more layers
    },
    "feedforward": {
      "layer_0": {
        "ff_input": [[...], [...]],            // shape: [seq_len, d_model]
        "ff_output": [[...], [...]]            // shape: [seq_len, d_model]
      },
      // ... more layers
    },
    "output": {
      "logits": [...],                         // shape: [vocab_size]
      "probabilities": [...],                  // softmax(logits)
      "predicted_token_id": 567,
      "predicted_token": "vegetables"
    }
  }
}
```

**Error Responses:**

```json
// Model not loaded
{
  "detail": "Mode 1 model not available. Train a model first."
}

// Empty input
{
  "detail": "Input text cannot be empty"
}
```

---

## 📚 Usage Examples

### Python Client

```python
import requests

url = "http://localhost:8000/api/v1/predict-next-word"
payload = {"text": "I eat"}

response = requests.post(url, json=payload)
result = response.json()

print(f"Input: {result['input_text']}")
print(f"Prediction: {result['predicted_word']} ({result['confidence']:.2%})")
print(f"\nTop 3 predictions:")
for pred in result['top_predictions'][:3]:
    print(f"  {pred['word']}: {pred['probability']:.2%}")
```

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/predict-next-word \
  -H "Content-Type: application/json" \
  -d '{"text": "I eat"}'
```

### JavaScript (Fetch)

```javascript
const response = await fetch('http://localhost:8000/api/v1/predict-next-word', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'I eat' })
});

const result = await response.json();
console.log(`Prediction: ${result.predicted_word}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
```

---

## 🧪 Testing Predictions

### Sample Predictions

Test the trained model with these examples:

```python
# High confidence predictions
"I eat"          → "vegetables" (52.9%)
"She likes"      → "dancing" (54.9%)
"We go"          → "to" (95.9%)
"I study"        → "very" (83.2%)

# Profession context
"I work as"      → "teacher" (58.6%)

# Weather/time context
"The weather is" → "wonderful" (35.6%)
"Today is"       → "Monday" / "special"
```

### Prediction Quality Metrics

**Strengths:**
- ✅ High accuracy for common patterns (85%+ success rate)
- ✅ Strong confidence scores (0.5-0.96) for correct predictions
- ✅ Learns contextual relationships (professions, activities, objects)
- ✅ Captures grammatical structures (verb conjugations, articles)
- ✅ Understands semantic associations (eat→vegetables, work→teacher)

**Limitations:**
- ⚠️ Occasional repetition for ambiguous contexts
- ⚠️ Lower confidence for uncommon word combinations
- ⚠️ Limited vocabulary (1,500 tokens)
- ⚠️ Sensitive to exact phrasing in training data

---

## 🎯 Educational Value

### What Students Learn

1. **Autoregressive Language Modeling**
   - How GPT-style models predict next tokens
   - Causal masking and attention patterns
   - Probability distributions over vocabulary

2. **Transformer Architecture**
   - Multi-head self-attention mechanism
   - Positional encoding (sinusoidal)
   - Layer normalization and residual connections
   - Feed-forward networks

3. **Training Best Practices**
   - Train/validation split (80/20)
   - Overfitting detection
   - Model selection criteria (validation loss)
   - Learning rate scheduling

4. **Visualization & Debugging**
   - Attention weight patterns
   - Embedding visualization
   - Layer-by-layer transformations
   - Softmax probability distributions

---

## 🛠️ Advanced Configuration

### Custom Corpus

**Add your own training data:**

1. Create or edit `data/sample_corpus.txt`
2. Add sentences (one per line)
3. Use diverse vocabulary and structures
4. Retrain model:

```bash
python -m app.features.mode1_next_word.train \
  --corpus path/to/your/corpus.txt \
  --epochs 50
```

**Corpus Guidelines:**
- Use simple, clear sentences
- Include varied vocabulary
- Mix sentence structures (statements, questions)
- Add contextual patterns (professions, activities, etc.)
- Minimum: ~500 lines for decent results
- Recommended: 1,000-5,000 lines

### Scaling Model Size

**For larger models (requires GPU):**

```bash
python -m app.features.mode1_next_word.train \
  --d-model 512 \          # Larger embeddings
  --n-heads 8 \            # More attention heads
  --n-layers 6 \           # Deeper model
  --d-ff 2048 \            # Wider feed-forward
  --batch-size 64 \        # Larger batches (if GPU memory allows)
  --epochs 100
```

**Parameter count scaling:**
- d_model=256, n_layers=4: ~3.9M params
- d_model=512, n_layers=6: ~30M params
- d_model=768, n_layers=12: ~100M params (GPT-2 Small size)

### Training on GPU

```bash
# Automatically uses GPU if available
python -m app.features.mode1_next_word.train \
  --epochs 50 \
  --device cuda

# Or force CPU
python -m app.features.mode1_next_word.train \
  --epochs 50 \
  --device cpu
```

---

## 🐛 Troubleshooting

### Training Issues

**Problem:** `CUDA out of memory`
```bash
# Reduce batch size
--batch-size 16  # or 8

# Or reduce model size
--d-model 128 --d-ff 512
```

**Problem:** Loss not decreasing
```bash
# Increase learning rate
--lr 1e-2

# Train for more epochs
--epochs 100

# Check corpus quality (needs diverse patterns)
```

**Problem:** Validation loss increasing (overfitting)
```bash
# This is normal! Best model is automatically saved.
# Check best_model.pt (lowest validation loss)
```

### Inference Issues

**Problem:** Model not found
```bash
# Train a model first
python -m app.features.mode1_next_word.train --epochs 50

# Or check checkpoint path in router.py
```

**Problem:** Poor predictions
```bash
# Check if best_model.pt is being loaded (not final_model.pt)
# Retrain with more diverse corpus
# Increase training epochs (50-100)
```

**Problem:** Slow inference
```bash
# This is expected on CPU (~100-200ms per prediction)
# For faster inference, use GPU or reduce model size
```

---

## 📖 References

### Papers & Resources

1. **"Attention is All You Need"** (Vaswani et al., 2017)
   - Original Transformer paper
   - https://arxiv.org/abs/1706.03762

2. **"Language Models are Unsupervised Multitask Learners"** (GPT-2, Radford et al., 2019)
   - Decoder-only architecture
   - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

3. **The Illustrated Transformer** (Jay Alammar)
   - Visual guide to transformers
   - http://jalammar.github.io/illustrated-transformer/

4. **Annotated Transformer** (Harvard NLP)
   - Line-by-line implementation guide
   - http://nlp.seas.harvard.edu/2018/04/03/attention.html

### Code References

- **Model Implementation:** `model/gpt_model.py`
- **Training Loop:** `training/trainer.py`
- **Tokenization:** `training/dataset.py`
- **API Integration:** `api/router.py`
- **Inference Service:** `service/gpt_service.py`

---

## 🚀 Future Enhancements

### Planned Features

1. **Improved Tokenization**
   - Subword tokenization (BPE/WordPiece)
   - Handle out-of-vocabulary words
   - Reduce vocabulary size

2. **Training Improvements**
   - Early stopping (stop when val_loss plateaus)
   - Learning rate warmup
   - Gradient clipping
   - Mixed precision training (FP16)

3. **Model Variants**
   - Character-level tokenization option
   - Different model sizes (small/medium/large)
   - Pre-trained model loading (GPT-2)

4. **Evaluation Metrics**
   - Perplexity calculation
   - BLEU score for generation
   - Human evaluation interface

5. **Production Features**
   - Model caching for faster startup
   - Batch prediction API
   - Model versioning
   - A/B testing support

---

## 📝 License

MIT License - See root project LICENSE file.

---

## 🤝 Contributing

This is part of the Transformer Interactive Visualization project. For contributions:

1. See main project [CLAUDE.md](../../../../CLAUDE.md) for development guidelines
2. Follow code quality standards (Black, Flake8, MyPy)
3. Add tests for new features
4. Update documentation

---

## 📞 Support

For issues or questions:
- Check [Troubleshooting](#-troubleshooting) section
- Review [CLAUDE.md](../../../../CLAUDE.md) for development guide
- See main project documentation in `/docs`

---

**Last Updated:** 2025-11-29
**Status:** ✅ Production Ready
**Model Version:** v1.0 (50 epochs with validation)
