# Transformer Visualization - Backend

FastAPI backend for transformer visualization platform. Features complete transformer implementations built from scratch with extensive educational documentation and visualization data extraction.

![Python](https://img.shields.io/badge/Python-3.9+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)

---

## 📋 Overview

The backend provides REST APIs for running transformer models and extracting visualization data for educational purposes. Built with a feature-based architecture for modularity and scalability.

### Key Features

- **From-Scratch Transformer Implementation** - Complete transformer with detailed educational comments
- **Visualization Data Extraction** - Attention weights, embeddings, activations at every layer
- **Feature-Based Architecture** - Self-contained modules for different transformer modes
- **REST API** - FastAPI endpoints for inference and visualization
- **Educational Focus** - Every component explained with mathematical foundations

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

#### Development

```bash
# Using Python module (recommended)
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Server URLs:**
- **API:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

**⚠️ Important:** Before using Mode 1, you need to train the model first:

```bash
# Train Mode 1 model (first time only, ~10-15 minutes)
python -m app.features.mode1_next_word.train --epochs 50
```

See [Mode 1 README](app/features/mode1_next_word/README.md) for detailed training instructions.

#### Production

```bash
# With Gunicorn + Uvicorn workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or with uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── features/                       # Feature modules (self-contained)
│   │   └── mode1_next_word/           # Mode 1: Next Word Prediction
│   │       ├── api/
│   │       │   └── router.py          # Mode 1 API routes
│   │       ├── model/
│   │       │   └── gpt_model.py       # GPT-style decoder transformer
│   │       ├── service/
│   │       │   └── gpt_service.py     # Inference and prediction
│   │       ├── training/
│   │       │   ├── dataset.py         # Tokenizer and data utilities
│   │       │   └── trainer.py         # Training loop with validation
│   │       ├── data/
│   │       │   └── sample_corpus.txt  # Training corpus
│   │       ├── checkpoints/
│   │       │   └── best_model.pt      # Trained model
│   │       ├── train.py               # Training script
│   │       └── README.md              # Mode 1 documentation
│   │
│   ├── models/                        # Core transformer components (shared)
│   │   ├── embeddings.py              # Token + positional embeddings
│   │   ├── attention.py               # Multi-head attention (CRITICAL)
│   │   ├── layers.py                  # Encoder/decoder layers
│   │   └── transformer.py             # Complete transformer model
│   │
│   ├── api/
│   │   └── routes.py                  # Main API routes + feature routers
│   │
│   └── main.py                        # FastAPI app entry point
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

### Architecture Principles

**Feature-Based Design:**
- Each mode (Mode 1, Mode 2, etc.) is a **self-contained module**
- All code, data, models, and training scripts in one folder
- Easy to add new features without affecting existing ones
- Clear separation of concerns

**Shared Components:**
- Core transformer building blocks in `app/models/`
- Can be reused across different features
- Educational focus with extensive documentation

---

## 🎯 Features

### Mode 1: Next Word Prediction (Mini-GPT)

**Status:** ✅ Production Ready

**Location:** `app/features/mode1_next_word/`

**Description:** GPT-style autoregressive language model that predicts the next word given input text. Includes complete training pipeline and visualization support.

**API Endpoint:**
```bash
POST /api/v1/predict-next-word
```

**Quick Example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict-next-word \
  -H "Content-Type: application/json" \
  -d '{"text": "I eat"}'
```

**Documentation:** See [Mode 1 README](app/features/mode1_next_word/README.md)

### Future Modes

**Planned Features:**
- **Mode 2:** Translation (Seq2Seq) - Full encoder-decoder transformer
- **Mode 3:** Masked Language Modeling (BERT-style)
- **Mode 4:** Custom Model Loading (GPT-2, BERT, etc.)

---

## 📡 API Endpoints

### Mode 1: Next Word Prediction

#### `POST /api/v1/predict-next-word`

Predict the next word given input text.

**Request:**
```json
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
    {"word": "breakfast", "probability": 0.287, "log_prob": -1.248}
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

**Visualization Data:**
The `steps` field contains complete visualization data for all 6 pipeline steps:
1. Tokenization
2. Embeddings + Positional Encoding
3. Self-Attention & Multi-Head Attention
4. Feedforward Network
5. Output Layer (Softmax)
6. Prediction Result

### Health & Info

#### `GET /`

Root endpoint with API information.

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-29T12:00:00"
}
```

#### `GET /api/v1/model/info`

Get model architecture information (future endpoint).

---

---

## 🛠️ Development

### Code Quality

```bash
# Format code (Black)
black app/

# Lint (Flake8)
flake8 app/

# Type checking (MyPy)
mypy app/

# Run all quality checks
black app/ && flake8 app/ && mypy app/
```

### Adding a New Feature Mode

1. **Create feature directory:**
   ```bash
   mkdir -p app/features/mode2_translation/{api,model,service,data,checkpoints}
   ```

2. **Implement components:**
   - `model/` - Transformer architecture
   - `service/` - Inference logic
   - `api/router.py` - API routes
   - `train.py` - Training script
   - `README.md` - Documentation

3. **Register router:**
   ```python
   # In app/api/routes.py
   from app.features.mode2_translation.api import router as mode2_router
   router.include_router(mode2_router, prefix="/api/v1", tags=["Mode 2"])
   ```

4. **Test and document:**
   - Add tests in `tests/test_features/`
   - Update main README
   - Update CLAUDE.md

### Adding a New API Endpoint

1. **Define route in feature's `api/router.py`:**
   ```python
   from fastapi import APIRouter
   from pydantic import BaseModel

   router = APIRouter()

   class PredictionRequest(BaseModel):
       text: str

   @router.post("/predict")
   async def predict(request: PredictionRequest):
       # Implementation
       return {"result": "..."}
   ```

2. **Add request/response models** using Pydantic

3. **Implement business logic** in service layer

4. **Add tests** in `tests/test_api/`

5. **Update documentation**

---

## 🎓 Educational Components

### Core Transformer Building Blocks

#### 1. Embeddings (`app/models/embeddings.py`)

**Token Embeddings + Positional Encoding**

```python
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformers.

    Why? Transformers have no inherent notion of position.
    Positional encoding injects position information.

    Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
             PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
```

#### 2. Attention (`app/models/attention.py`)

**Scaled Dot-Product Attention + Multi-Head Attention**

```python
class ScaledDotProductAttention(nn.Module):
    """
    Core attention mechanism.

    Formula: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Why scaling? Large dot products push softmax into regions
    with tiny gradients. Scaling by sqrt(d_k) keeps values stable.
    """
```

#### 3. Layers (`app/models/layers.py`)

**Feed-Forward Networks, Encoder/Decoder Layers**

```python
class TransformerEncoderLayer(nn.Module):
    """
    Single encoder layer: Self-Attention + FFN

    Flow:
    1. Multi-Head Self-Attention
    2. Add & Norm (residual + layer norm)
    3. Feed-Forward Network
    4. Add & Norm
    """
```

#### 4. Complete Transformer (`app/models/transformer.py`)

**Full encoder-decoder architecture**

### Visualization Data Structure

All forward passes return `(output, viz_data)` tuple:

```python
output, viz_data = model(src, tgt)

# viz_data contains:
{
    'attention_weights': [...],    # Each layer, each head
    'embeddings': [...],           # Token + positional
    'layer_outputs': [...],        # Each layer's output
    'activation_stats': {...}      # Min, max, mean, std
}
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```env
# Server
HOST=0.0.0.0
PORT=8000

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Model
MODEL_DEVICE=cuda  # or cpu
MODEL_CACHE_DIR=./checkpoints

# Logging
LOG_LEVEL=INFO
```

### Model Configuration

Adjust model size in feature modules (e.g., `app/features/mode1_next_word/train.py`):

```python
model = GPTModel(
    vocab_size=1500,
    d_model=256,        # Increase for larger model
    n_heads=4,          # Must divide d_model
    n_layers=4,         # More layers = deeper model
    d_ff=1024,          # Feed-forward dimension
    max_len=60,
    dropout=0.1
)
```

---

## 🐛 Troubleshooting

### Common Issues

#### CORS Errors

**Problem:** Frontend can't connect to backend

**Solution:** Update CORS settings in `app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React dev server
        "http://localhost:5173",   # Vite dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'app'`

**Solution:**
- Ensure you're in `backend/` directory
- Virtual environment is activated
- Run as module: `python -m app.main`

#### Memory Issues

**Problem:** `CUDA out of memory` or slow CPU inference

**Solution:**
```python
# Reduce model size
d_model=128  # Smaller dimension
n_layers=2   # Fewer layers
batch_size=16  # Smaller batches
```

#### Model Not Found

**Problem:** Mode 1 model not loading

**Solution:**
```bash
# Train a model first
cd backend
python -m app.features.mode1_next_word.train --epochs 50

# Check checkpoint exists
ls app/features/mode1_next_word/checkpoints/best_model.pt
```

---

## 🚀 Deployment

### Docker

**Dockerfile:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**

```bash
# Build image
docker build -t transformer-backend .

# Run container
docker run -p 8000:8000 transformer-backend

# With GPU support
docker run --gpus all -p 8000:8000 transformer-backend
```

### Docker Compose

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - MODEL_DEVICE=cuda
    volumes:
      - ./backend/app/features:/app/app/features
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Production Checklist

- [ ] Use Gunicorn with Uvicorn workers
- [ ] Enable HTTPS (reverse proxy with Nginx/Caddy)
- [ ] Set up logging and monitoring
- [ ] Configure CORS for production domain
- [ ] Use environment variables for secrets
- [ ] Set up health check endpoints
- [ ] Enable request validation
- [ ] Implement rate limiting
- [ ] Set up model caching
- [ ] Configure automatic restarts

---

## 📊 Performance

### Benchmarks

**Mode 1 (Next Word Prediction):**
- Model initialization: ~500ms (CPU), ~200ms (GPU)
- Inference (single prediction): ~100-200ms (CPU), ~10-20ms (GPU)
- Batch inference (32 samples): ~1-2s (CPU), ~100-200ms (GPU)
- Visualization data extraction: ~50ms

**Optimization Tips:**
1. **Model Caching** - Load model once, reuse for all requests
2. **Batch Processing** - Process multiple requests together
3. **GPU Acceleration** - Use CUDA for faster inference
4. **Response Compression** - Gzip responses (built into FastAPI)
5. **Async Processing** - Use async/await for I/O operations

---

## 📚 Documentation

### API Documentation

- **Interactive Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

### Code Documentation

Each module includes extensive docstrings:

```python
"""
Module Description

This module implements [component] with [features].

Key Classes:
    - ClassName: Description

Educational Notes:
    - Mathematical foundations
    - Why this approach
    - Common pitfalls
"""
```

### Additional Resources

- **Mode 1 Documentation:** [app/features/mode1_next_word/README.md](app/features/mode1_next_word/README.md)
- **Frontend Documentation:** [../frontend/README.md](../frontend/README.md)
- **Developer Guide:** [../CLAUDE.md](../CLAUDE.md)
- **Project Plan:** [../PROJECT_PLAN.md](../PROJECT_PLAN.md)

---

## 🤝 Contributing

### Development Workflow

1. **Create feature branch:**
   ```bash
   git checkout -b feature/new-mode
   ```

2. **Develop feature:**
   - Follow feature-based architecture
   - Add comprehensive docstrings
   - Include educational explanations

3. **Quality checks:**
   ```bash
   black app/
   flake8 app/
   mypy app/
   pytest
   ```

4. **Update documentation:**
   - Feature README
   - API docs
   - Main README
   - CLAUDE.md

5. **Submit pull request**

### Code Style

**Python:**
- **Formatter:** Black (line length 100)
- **Linter:** Flake8
- **Type Checker:** MyPy (strict mode)
- **Docstrings:** Google style with educational focus

**Educational Philosophy:**
Every component should answer:
- **What:** What does this do?
- **Why:** Why is it needed?
- **How:** How does it work mathematically?

---

## 📜 License

MIT License - See [LICENSE](../LICENSE) file for details.

---

## 📞 Support

For issues, questions, or contributions:

- **Issues:** Check troubleshooting section above
- **Developer Guide:** [CLAUDE.md](../CLAUDE.md)
- **Mode 1 Docs:** [app/features/mode1_next_word/README.md](app/features/mode1_next_word/README.md)
- **Frontend Docs:** [../frontend/README.md](../frontend/README.md)

---

**Last Updated:** 2025-11-29
**Status:** ✅ Production Ready
**Current Features:** Mode 1 (Next Word Prediction)
