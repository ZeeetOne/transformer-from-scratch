# Transformer Visualization - Backend

FastAPI backend for transformer visualization platform. Implements transformer from scratch with extensive educational documentation and visualization data extraction.

## Features

- **From-Scratch Implementation**: Complete transformer with detailed comments
- **Visualization Data**: Extract attention weights, embeddings, activations
- **REST API**: FastAPI endpoints for inference and visualization
- **Educational Focus**: Every component explained for learning

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Server

### Development

```bash
# Using Python module
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at:
- API: http://localhost:8000
- Docs (Swagger): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Architecture

### Core Components

#### 1. Models (`app/models/`)

- **embeddings.py**: Token embeddings + positional encoding
- **attention.py**: Scaled dot-product attention + multi-head attention
- **layers.py**: Feed-forward networks, encoder/decoder layers
- **transformer.py**: Complete transformer model

#### 2. Services (`app/services/`)

- **inference.py**: Model inference and tokenization
- **visualization.py**: Extract and format visualization data

#### 3. API (`app/api/`)

- **routes.py**: REST API endpoints

## API Endpoints

### Inference

```bash
POST /api/v1/inference
```

Request:
```json
{
  "source_text": "Hello world",
  "generate": true,
  "max_gen_len": 20
}
```

Response includes:
- Decoded output
- Token IDs
- Complete visualization data

### Attention Visualization

```bash
POST /api/v1/attention
```

Get attention heatmap data for specific layer/head.

### Model Info

```bash
GET /api/v1/model/info
```

Returns model architecture specifications.

## Model Configuration

Default configuration (educational size):

```python
TransformerService(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=256,           # Model dimension
    n_heads=4,             # Attention heads
    n_encoder_layers=2,    # Encoder layers
    n_decoder_layers=2,    # Decoder layers
    d_ff=1024,            # Feed-forward dimension
    dropout=0.1
)
```

Adjust in `app/api/routes.py` for different model sizes.

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app tests/

# Specific test file
pytest tests/test_transformer.py
```

## Development

### Code Quality

```bash
# Format code
black app/

# Lint
flake8 app/

# Type checking
mypy app/
```

### Adding New Endpoints

1. Define route in `app/api/routes.py`
2. Add request/response models using Pydantic
3. Implement business logic in services
4. Add tests in `tests/`

## Educational Notes

### Transformer Implementation

Each component includes extensive documentation:

```python
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention: The fundamental attention mechanism.

    Formula: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Why scaling? Large dot products push softmax into regions
    with tiny gradients. Scaling by sqrt(d_k) keeps values
    in a reasonable range.
    """
```

### Visualization Data Structure

All forward passes return tuple: `(output, viz_data)`

```python
output, viz_data = model(src, tgt)
# viz_data contains:
# - attention_weights for each head
# - embeddings at each stage
# - layer-wise outputs
# - activation statistics
```

## Troubleshooting

### CORS Issues

Ensure frontend URL is in CORS allowed origins (`app/main.py`):

```python
allow_origins=[
    "http://localhost:3000",
    "http://localhost:5173",
]
```

### Memory Issues

Reduce model size or batch size:

```python
d_model=128,  # Smaller dimension
n_layers=1,   # Fewer layers
```

### Import Errors

Ensure you're in the backend directory and virtual environment is activated.

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t transformer-viz-backend .
docker run -p 8000:8000 transformer-viz-backend
```

## Performance

- Model initialization: ~500ms
- Inference (10 tokens): ~100-200ms
- Visualization data extraction: ~50ms

For production, consider:
- Model caching
- Batch processing
- GPU acceleration
- Response compression

## License

MIT
