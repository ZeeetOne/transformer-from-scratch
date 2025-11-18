# Training Guide: GPT-Style Language Model from Scratch

This guide explains how to train your own Mini-GPT language model from scratch using this platform.

## Overview

This project implements a **complete end-to-end GPT-style transformer** that learns next-word prediction from raw text. Unlike the main visualization tool (which uses encoder-decoder architecture), Mode 1 uses a **decoder-only architecture** similar to GPT models.

## Training Pipeline

### 1. Data Preparation
- **Input**: A text corpus (.txt file)
- **Tokenization**: Character-level or word-level
- **Dataset**: Converts text into (input, target) pairs using **shifted target technique**
  - Input: `[token1, token2, token3, token4]`
  - Target: `[token2, token3, token4, token5]`
  - Each token predicts the next token

### 2. Model Architecture
- **Embedding Layer**: Converts tokens to dense vectors
- **Positional Encoding**: Adds position information
- **Transformer Blocks** (×N layers):
  - Self-Attention (masked, causal)
  - Multi-Head Attention
  - Feed-Forward Network
  - Residual Connections
  - Layer Normalization
- **Output Layer**: Linear projection + Softmax → probability distribution

### 3. Training Process
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam optimizer
- **Backpropagation**: Updates all weights
- **Gradient Clipping**: Prevents exploding gradients

### 4. Inference
- After training, the model can predict next words **without needing the dataset**
- Only requires:
  - Trained weights (checkpoint file)
  - Tokenizer vocabulary

## Quick Start

### Step 1: Train Your Model

```bash
cd backend
python train_gpt_model.py --corpus data/sample_corpus.txt --epochs 50
```

**Training Parameters:**
```bash
python train_gpt_model.py \
  --corpus data/sample_corpus.txt \
  --level char \              # char or word tokenization
  --epochs 50 \               # number of training epochs
  --batch-size 32 \           # batch size
  --lr 0.001 \                # learning rate
  --d-model 256 \             # model dimension
  --n-heads 4 \               # attention heads
  --n-layers 4 \              # transformer layers
  --d-ff 1024 \               # feed-forward dimension
  --max-seq-len 50            # maximum sequence length
```

### Step 2: Monitor Training

Training output will show:
```
Epoch 1/50
----------------------------------------------------------
  Batch 0/100, Loss: 4.5234
  Batch 10/100, Loss: 4.1234
  ...

Epoch 1 Summary:
  Train Loss: 3.8234
  Time: 15.23s
  ✓ Best model saved (val_loss: 3.8234)
```

### Step 3: Use Trained Model

The trained model is automatically saved to:
- `checkpoints/best_model.pt` - Best model (lowest loss)
- `checkpoints/final_model.pt` - Last epoch
- `checkpoints/checkpoint_epoch_N.pt` - Periodic checkpoints

### Step 4: Run Inference

The API server **automatically loads** `checkpoints/best_model.pt` if it exists:

```bash
# Start backend
cd backend
python -m app.main
```

You'll see:
```
Loading trained model from: checkpoints/best_model.pt
✓ Loaded checkpoint (epoch 49, loss: 1.2345)
✓ Vocabulary size: 89
```

Now visit Mode 1 in the web interface, and predictions will use your trained model!

## Understanding the Training Process

### What Happens During Training?

1. **Epoch 1-10**: Model learns basic patterns
   - Loss decreases rapidly
   - Model starts recognizing common tokens
   - Predictions are still mostly random

2. **Epoch 10-30**: Model learns relationships
   - Loss decreases steadily
   - Model learns word associations (e.g., "I" → "eat")
   - Predictions become more coherent

3. **Epoch 30-50**: Model refines predictions
   - Loss decreases slowly
   - Model learns context-dependent patterns
   - Predictions become meaningful

### Example Training Results

**Before Training (Random):**
```
Input: "I eat"
Prediction: "<UNK>" (nonsense)
Confidence: 0.5%
```

**After Training (50 epochs):**
```
Input: "I eat"
Prediction: "rice" or "breakfast"
Confidence: 15-25%
```

## Creating Your Own Corpus

### Option 1: Use Sample Corpus (Recommended for Testing)
The included `data/sample_corpus.txt` contains ~500 simple English sentences focused on common patterns.

### Option 2: Create Custom Corpus

Create a `.txt` file with your text:

```
# my_corpus.txt
The quick brown fox jumps over the lazy dog.
Machine learning is transforming technology.
Transformers use attention mechanisms.
...
```

**Best Practices:**
- **Size**: At least 10,000 words for meaningful learning
- **Quality**: Clean, grammatically correct text
- **Patterns**: Repetitive patterns help the model learn
- **Domain**: Focus on specific domain for better results

**Train with custom corpus:**
```bash
python train_gpt_model.py --corpus my_corpus.txt --epochs 100
```

## Tokenization Strategies

### Character-Level Tokenization (Default)
```bash
python train_gpt_model.py --level char
```

**Pros:**
- Small vocabulary (<100 tokens)
- No out-of-vocabulary issues
- Works with any language

**Cons:**
- Longer sequences
- Harder to learn long-range dependencies

**Best For:**
- Small datasets
- Learning basic patterns
- Quick experiments

### Word-Level Tokenization
```bash
python train_gpt_model.py --level word
```

**Pros:**
- Shorter sequences
- Better semantic understanding
- Faster training

**Cons:**
- Larger vocabulary (1000s of tokens)
- Out-of-vocabulary words become `<UNK>`

**Best For:**
- Larger datasets
- Real language modeling
- Production use

## Model Architecture Details

### Default Configuration
```python
d_model = 256        # Embedding dimension
n_heads = 4          # Attention heads per layer
n_layers = 4         # Transformer layers
d_ff = 1024          # Feed-forward hidden dimension
max_seq_len = 50     # Maximum sequence length
```

**Total Parameters**: ~1.5M parameters (with vocab_size=100)

### Scaling Up

For better performance with more data:
```bash
python train_gpt_model.py \
  --d-model 512 \
  --n-heads 8 \
  --n-layers 6 \
  --d-ff 2048 \
  --epochs 100
```

**Warning**: Larger models:
- Take longer to train
- Require more memory
- Need more data to avoid overfitting

## Troubleshooting

### Issue: Loss not decreasing
**Solutions:**
- Increase training epochs
- Reduce learning rate: `--lr 0.0001`
- Check corpus quality
- Increase model size

### Issue: Loss decreasing too slowly
**Solutions:**
- Increase learning rate: `--lr 0.01`
- Reduce model size for small datasets
- Use word-level tokenization

### Issue: Model overfitting
**Symptoms**: Training loss very low, but predictions poor
**Solutions:**
- Add more training data
- Reduce model size
- Increase dropout rate (in code)

### Issue: Out of memory
**Solutions:**
- Reduce batch size: `--batch-size 16`
- Reduce model size: `--d-model 128 --n-layers 2`
- Reduce sequence length: `--max-seq-len 30`

### Issue: Predictions still showing `<UNK>`
**Cause**: Token not in vocabulary
**Solutions:**
- Use character-level tokenization
- Expand vocabulary in training corpus
- Check if model loaded correctly

## Advanced Usage

### Resume Training from Checkpoint

Edit `train_gpt_model.py` to load a checkpoint:
```python
# Before training
trainer.load_checkpoint('checkpoints/checkpoint_epoch_20.pt')
```

### Generate Long Text

After training, test generation:
```python
from app.services.gpt_service import GPTService

gpt = GPTService(checkpoint_path='checkpoints/best_model.pt')

# Generate text
input_text = "I eat"
tokens = gpt.tokenize(input_text, add_special_tokens=True)
# ... implement generation loop ...
```

### Export Model for Production

The checkpoint file contains everything needed:
- Model weights
- Tokenizer vocabulary
- Model configuration

Simply copy `best_model.pt` to your production environment.

## Expected Training Times

**Small Corpus (10K words, 50 epochs):**
- CPU: ~10-15 minutes
- GPU: ~2-3 minutes

**Medium Corpus (100K words, 100 epochs):**
- CPU: ~2-3 hours
- GPU: ~20-30 minutes

**Large Corpus (1M words, 200 epochs):**
- CPU: ~1-2 days
- GPU: ~3-4 hours

## Understanding Loss Values

### Cross-Entropy Loss
- **Initial (untrained)**: 4-5 (random predictions)
- **After 10 epochs**: 2-3 (learning patterns)
- **After 50 epochs**: 1-2 (good predictions)
- **Well-trained**: 0.5-1.0 (very good)

### Perplexity
Perplexity = exp(loss)
- Lower is better
- Perplexity of 2-3 is excellent for small models

## Example Workflow

### 1. Quick Test (5 minutes)
```bash
python train_gpt_model.py --epochs 10
```

### 2. Standard Training (30 minutes)
```bash
python train_gpt_model.py --epochs 50 --lr 0.001
```

### 3. Production Model (2 hours)
```bash
python train_gpt_model.py \
  --corpus my_large_corpus.txt \
  --level word \
  --epochs 200 \
  --batch-size 64 \
  --d-model 512 \
  --n-heads 8 \
  --n-layers 6
```

## Files and Directories

```
backend/
├── train_gpt_model.py          # Training script
├── data/
│   └── sample_corpus.txt       # Sample training data
├── checkpoints/                # Saved models (created during training)
│   ├── best_model.pt          # Best model (auto-loaded by API)
│   ├── final_model.pt         # Final epoch
│   └── checkpoint_epoch_*.pt  # Periodic checkpoints
└── app/
    ├── services/
    │   └── gpt_service.py     # GPT model + inference
    └── training/
        ├── dataset.py         # Data preparation
        └── trainer.py         # Training loop
```

## Next Steps

1. **Train a model** on the sample corpus
2. **Test predictions** in Mode 1
3. **Create your own corpus** for domain-specific predictions
4. **Experiment with hyperparameters** for better results
5. **Scale up** for production use

## Educational Value

This project demonstrates:
- ✅ **Complete end-to-end ML pipeline** (data → training → inference)
- ✅ **Transformer architecture** (attention, embeddings, positional encoding)
- ✅ **Language modeling** (next-word prediction)
- ✅ **Training from scratch** (no pre-trained models)
- ✅ **Production deployment** (API integration)

Perfect for learning how GPT-style models work under the hood!

## Support

For issues or questions:
- Check troubleshooting section above
- Review training logs for errors
- Verify corpus format and quality
- Try with sample corpus first

---

**Remember**: The model learns from your data. Quality input = quality predictions!
