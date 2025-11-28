# Transformer Interactive Visualization

> An educational platform for understanding transformer architecture through interactive, step-by-step visualizations.

![Architecture](https://img.shields.io/badge/Architecture-Transformer-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)
![React](https://img.shields.io/badge/React-18.2-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)

---

## ✨ What is This?

This project helps you **understand transformers** - the AI technology behind ChatGPT, BERT, and modern language models. Instead of reading complex papers, you can:

✅ **Visualize** how transformers process text step-by-step
✅ **Interact** with real models trained from scratch
✅ **Learn** the math and intuition behind attention mechanisms
✅ **Experiment** with different inputs and see real-time results

Perfect for:
- 🎓 **Students** learning about transformers
- 👨‍💻 **ML Engineers** wanting to understand internals
- 🧑‍🏫 **Teachers** explaining transformers visually
- 🔬 **Researchers** prototyping transformer variants

---

## 🎯 What Can You Do?

### Mode 1: Next Word Prediction (Mini-GPT)

Train and visualize a GPT-style model that predicts the next word in a sentence.

**Example:**
- Input: "I eat"
- Output: "vegetables" (52.9% confidence)

**6-Step Visualization Pipeline:**

1. **Tokenization** - See how text becomes tokens
2. **Embeddings** - Understand semantic meaning + position encoding
3. **Attention** - Watch how words "attend" to each other
4. **Feedforward** - See neural network transformations
5. **Softmax** - Probability distribution over vocabulary
6. **Prediction** - Final result with confidence scores

Each step shows:
- 📊 Visual representations (heatmaps, graphs, grids)
- 🧮 Mathematical formulas
- 💡 Educational annotations
- 🔍 Interactive exploration

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites

Before you begin, make sure you have:
- ✅ **Python 3.9+** - [Download here](https://www.python.org/downloads/)
- ✅ **Node.js 16+** - [Download here](https://nodejs.org/)

---

### Step 1: Backend Setup

Open a terminal and run:

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
python -m app.main
```

**✅ Success!** Backend is running at http://localhost:8000

**🔍 Verify:** Open http://localhost:8000/docs in your browser - you should see the API documentation.

---

### Step 2: Train Mode 1 Model (First Time Only)

**⚠️ Important:** You need to train the model before using it!

Open a **new terminal** (keep backend running) and run:

```bash
# Navigate to backend (make sure venv is activated)
cd backend
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Train Mode 1 model (takes ~10-15 minutes on CPU)
python -m app.features.mode1_next_word.train --epochs 50

# Wait for training to complete...
# You'll see: "TRAINING COMPLETE!" when done
# Model saved to: backend/app/features/mode1_next_word/checkpoints/best_model.pt
```

**What happens:**
- Trains GPT-style model on sample corpus (1,449 lines)
- 50 epochs with 80/20 train/validation split
- Saves best model automatically
- Shows training progress and loss curves

**✅ Success!** Model trained and saved to `checkpoints/best_model.pt`

**🔍 Verify:** Check that `backend/app/features/mode1_next_word/checkpoints/best_model.pt` exists

**💡 Tip:** You only need to do this once. The trained model will be reused on backend restart.

---

### Step 3: Frontend Setup

Open a **new terminal** (keep backend running) and run:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start frontend server
npm run dev
```

**✅ Success!** Frontend is running at http://localhost:3000

**🔍 Verify:** Open http://localhost:3000 in your browser - you should see the landing page.

---

### Step 4: Try Mode 1

1. **Open** http://localhost:3000 in your browser
2. **Click** "Applications" → "Mode 1: Next Word Prediction"
3. **Enter** text like "I eat" or "She likes"
4. **Click** "Predict Next Word"
5. **Explore** the 6 visualization steps!

---

## 📚 User Guide

### Using Mode 1: Next Word Prediction

#### Input Examples to Try:

```
"I eat"           → vegetables, breakfast, rice
"She likes"       → dancing, music, chocolate
"We go"           → to, home, school
"The weather is"  → wonderful, nice, cold
"I work as"       → teacher, engineer, doctor
```

#### Understanding the Visualizations:

**Step 1: Tokenization**
- Shows how your text is split into tokens (words)
- Each token gets a unique ID number
- Color-coded for easy tracking

**Step 2: Embeddings + Positional Encoding**
- **Word Embeddings:** Shows semantic meaning (similar words have similar patterns)
- **Positional Encoding:** Shows position information (sinusoidal waves)
- **Final:** Combined embedding used by the model

**Step 3: Attention**
- **Q/K/V Projections:** Shows how input is transformed
- **Attention Weights:** Which words are "looking at" which words
- **Multi-Head:** Model uses 4 different attention heads
- Interactive head selector to explore different attention patterns

**Step 4: Feedforward Network**
- Shows dimension expansion (256 → 1024 → 256)
- ReLU activation function visualization
- Input/output comparison

**Step 5: Softmax Output**
- Probability distribution over all possible next words
- Top-10 predictions with confidence scores
- Bar chart visualization

**Step 6: Prediction Result**
- Final predicted word
- Confidence percentage
- Alternative predictions

---

## 🛠️ Advanced: Train Your Own Model

Want to train a custom model on your own text?

### Step 1: Prepare Your Corpus

Create a text file with sentences (one per line):

```
backend/app/features/mode1_next_word/data/my_corpus.txt
```

Example content:
```
I love programming.
Python is a great language.
Transformers are powerful models.
...
```

**Tips:**
- Minimum: ~500 lines for decent results
- Recommended: 1,000-5,000 lines
- Use simple, clear sentences
- Mix different sentence structures

### Step 2: Train the Model

```bash
# Navigate to backend (with venv activated)
cd backend

# Train for 50 epochs
python -m app.features.mode1_next_word.train \
  --corpus app/features/mode1_next_word/data/my_corpus.txt \
  --epochs 50

# Training will take ~10-15 minutes on CPU
```

**What happens:**
- ✅ Tokenizes your corpus
- ✅ Builds vocabulary
- ✅ Trains for 50 epochs with validation
- ✅ Saves `best_model.pt` (lowest validation loss)
- ✅ Shows training progress and loss curves

### Step 3: Use Your Model

The API automatically loads `best_model.pt` on startup. Just restart the backend:

```bash
# Stop backend (Ctrl+C)
# Start again
python -m app.main
```

Your custom model is now being used!

**Training Options:**

```bash
# More epochs (better quality, takes longer)
--epochs 100

# Larger model (more parameters, slower)
--d-model 512 --n-heads 8 --n-layers 6

# Custom learning rate
--lr 1e-3

# Use GPU (if available)
--device cuda
```

---

## 📊 Understanding Model Quality

### Training Loss vs Validation Loss

**Training Loss:** How well model fits training data
**Validation Loss:** How well model generalizes to new data ⭐

✅ **Best model:** Saved at epoch with **lowest validation loss**
❌ **Don't use:** Final epoch model (may be overfit)

**Example:**
```
Epoch 6:  train_loss=5.5,  val_loss=4.69  ← BEST (saved as best_model.pt)
Epoch 50: train_loss=0.99, val_loss=5.38  ← Overfit (don't use)
```

**Why is validation loss higher?**
- It's measured on **unseen data** (realistic performance)
- Lower training loss doesn't mean better model!
- Validation loss is the **true** measure of quality

See [VALIDATION_COMPARISON.md](VALIDATION_COMPARISON.md) for detailed explanation.

---

## 🎓 Learning Resources

### Transformer Papers & Guides

- **[Attention is All You Need](https://arxiv.org/abs/1706.03762)** - Original Transformer paper (Vaswani et al., 2017)
- **[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)** - Visual guide by Jay Alammar
- **[Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)** - Harvard NLP code walkthrough

### How This Project Helps

Unlike reading papers, this project lets you:
- ✅ **See** attention weights in real-time
- ✅ **Experiment** with different inputs
- ✅ **Understand** the math step-by-step
- ✅ **Train** your own models from scratch

---

## 🔧 Troubleshooting

### Backend won't start

**Problem:** `ModuleNotFoundError` or import errors

**Solution:**
```bash
# Make sure you're in backend/ directory
cd backend

# Make sure virtual environment is activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend won't connect to backend

**Problem:** "Network Error" or "Failed to fetch"

**Solution:**
1. Check backend is running (http://localhost:8000/docs should work)
2. Check no firewall blocking port 8000
3. Try restarting both backend and frontend

### Model predictions are random/bad

**Problem:** Model predicts nonsense

**Solution:**
- ✅ Check `best_model.pt` exists in `backend/app/features/mode1_next_word/checkpoints/`
- ✅ Train model for more epochs (50-100)
- ✅ Expand training corpus (more diverse sentences)
- ✅ Try different input phrases (model learns from training data)

### Training is slow

**Problem:** Training takes too long

**Solution:**
- ✅ Use GPU: `--device cuda` (if available)
- ✅ Reduce model size: `--d-model 128 --n-layers 2`
- ✅ Reduce epochs: `--epochs 20` (quick test)
- ℹ️ Normal: ~10-15 minutes for 50 epochs on CPU

---

## 📖 Documentation

- **[Backend Guide](backend/README.md)** - Backend setup, API, development
- **[Frontend Guide](frontend/README.md)** - Frontend setup, components, UI
- **[Mode 1 Complete Guide](backend/app/features/mode1_next_word/README.md)** - Training, inference, API
- **[Developer Guide (CLAUDE.md)](CLAUDE.md)** - For contributors and developers
- **[Project Plan](PROJECT_PLAN.md)** - Original vision and technical decisions

---

## 🚧 Roadmap

### Current Status

✅ **Mode 1: Next Word Prediction** - Production ready
- GPT-style decoder-only transformer
- 6-step visualization pipeline
- Training from scratch
- Interactive exploration

### Coming Soon

🔜 **Mode 2: Translation (Seq2Seq)**
- Full encoder-decoder architecture
- Translate between languages
- Visualize encoder-decoder attention

🔜 **Mode 3: Masked Language Modeling (BERT-style)**
- Bidirectional attention
- Fill in the blanks
- Sentence understanding

🔜 **Mode 4: Load Pre-trained Models**
- Load GPT-2, BERT, etc.
- Visualize production models
- Compare architectures

### UI Enhancements

- Dark mode
- Export visualizations (PNG/SVG)
- Animation playback controls
- Comparison mode (side-by-side inputs)

---

## 🤝 Contributing

Contributions are welcome! This is an **educational project** focused on clarity and learning.

**How to contribute:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-visualization`)
3. Make your changes
4. Test thoroughly
5. Submit a pull request

**Guidelines:**
- Maintain educational focus (clear explanations)
- Add comments explaining **what**, **why**, **how**
- Include visual examples if adding visualizations
- Update relevant documentation

See [CLAUDE.md](CLAUDE.md) for developer guide.

---

## 📜 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- **Vaswani et al.** for "Attention is All You Need" (2017)
- **Jay Alammar** for "The Illustrated Transformer"
- **Harvard NLP** for "The Annotated Transformer"
- **PyTorch** and **FastAPI** communities
- All contributors and users of this project

---

## 📞 Support & Feedback

**Have questions or found a bug?**

- 📖 Check the [troubleshooting section](#-troubleshooting) above
- 📚 Read the [documentation](#-documentation)
- 🐛 [Open an issue](https://github.com/your-username/transformer-from-scratch/issues) on GitHub
- 💬 Start a discussion for feature requests

**Want to learn more about transformers?**

- Start with Mode 1 and explore all 6 steps
- Try different input texts and observe patterns
- Train your own model on custom data
- Read the papers listed in [Learning Resources](#-learning-resources)

---

**Built with educational clarity in mind** - Helping people understand transformers through interactive visualization.

**⭐ If this helped you understand transformers, please star the repo!**

---

**Last Updated:** 2025-11-29
**Status:** ✅ Production Ready (Mode 1)
**Version:** 1.0
