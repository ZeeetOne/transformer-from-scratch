# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Transformer Interactive Visualization** - An educational platform for learning transformer architecture through interactive visualizations. Built from scratch with detailed explanations at every stage.

**Purpose**: Help ML practitioners, students, and engineers understand transformers by visualizing embeddings, attention mechanisms, and complete architecture.

**Architecture**: Full-stack application with Python/FastAPI backend and React/TypeScript frontend.

## Technology Stack

### Backend
- **Language**: Python 3.9+
- **Framework**: FastAPI (async web framework)
- **ML**: PyTorch (transformer implementation from scratch)
- **Libraries**: NumPy, Pydantic

### Frontend
- **Language**: TypeScript
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Visualizations**: Plotly.js, D3.js
- **State**: React hooks (useState, useMemo)

## Development Commands

### Backend

```bash
# Navigate to backend
cd backend

# Create and activate virtual environment (first time)
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python -m app.main
# Or: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Code quality
black app/              # Format
flake8 app/            # Lint
mypy app/              # Type checking
```

Backend runs at: http://localhost:8000
API docs: http://localhost:8000/docs

### Frontend

```bash
# Navigate to frontend
cd frontend

# Install dependencies (first time)
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint
npm run lint
```

Frontend runs at: http://localhost:3000

### Running Both Simultaneously

Open two terminal windows:
1. Terminal 1: `cd backend && python -m app.main`
2. Terminal 2: `cd frontend && npm run dev`

## Code Architecture

### Backend Structure

```
backend/app/
├── models/                    # Transformer implementation
│   ├── embeddings.py         # Token + positional embeddings
│   ├── attention.py          # Multi-head attention (CRITICAL)
│   ├── layers.py             # Encoder/decoder layers
│   └── transformer.py        # Complete model
├── services/
│   ├── inference.py          # Model inference + tokenization
│   └── visualization.py      # Extract viz data
├── api/
│   └── routes.py             # REST API endpoints
└── main.py                   # FastAPI app entry
```

**Key Pattern**: All model forward passes return `(output, viz_data)` tuple for visualization.

**Educational Focus**: Every component has extensive docstrings explaining the math and intuition.

### Frontend Structure

```
frontend/src/
├── components/
│   ├── ControlPanel.tsx           # Input controls
│   ├── AttentionVisualizer.tsx    # Attention heatmaps
│   ├── EmbeddingVisualizer.tsx    # Embedding plots
│   └── ArchitectureDiagram.tsx    # Architecture view
├── services/
│   └── api.ts                     # Backend API client
├── App.tsx                        # Main app
└── main.tsx                       # Entry point
```

**Key Pattern**: Components receive `InferenceResponse` prop and extract visualization data using `useMemo`.

## Important Implementation Details

### Transformer Model

**Configuration** (educational size):
- d_model: 256 (model dimension)
- n_heads: 4 (attention heads)
- n_encoder_layers: 2
- n_decoder_layers: 2
- d_ff: 1024 (feed-forward dimension)

**Location**: Backend model size configured in `backend/app/api/routes.py` (transformer_service initialization)

**Scaling**: All embeddings scaled by √d_model (Attention is All You Need paper)

### Attention Mechanism

**Location**: `backend/app/models/attention.py`

**Critical Components**:
1. `ScaledDotProductAttention`: Core attention (Q, K, V)
2. `MultiHeadAttention`: Parallel heads with projections

**Formula**: `Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V`

**Visualization Data**: Each head returns attention_weights array for heatmap rendering

### API Endpoints

**Main Endpoints** (`/api/v1/...`):
- `POST /inference` - Run transformer, get complete viz data
- `POST /attention` - Get attention heatmaps for layer/head
- `GET /model/info` - Model architecture specs
- `POST /visualize/embeddings` - Embedding progression
- `POST /visualize/flow` - Attention flow (token connections)
- `POST /visualize/complete` - All viz data formatted for frontend

### Tokenization

**Current**: Simple character-level tokenizer (demo purposes)
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- Location: `backend/app/services/inference.py`

**Future**: Can swap for tiktoken, sentencepiece, or HuggingFace tokenizer

### Visualization Data Flow

1. User enters text in ControlPanel
2. Frontend calls `/api/v1/inference` endpoint
3. Backend runs forward pass, extracts viz data at each layer
4. Frontend receives data, components extract relevant parts
5. Plotly/D3 render interactive visualizations

## Common Development Tasks

### Adding a New Visualization

1. **Backend**: Extract data in `backend/app/services/visualization.py`
   ```python
   @staticmethod
   def extract_new_viz(viz_data: Dict) -> Dict:
       # Extract and format data
       return formatted_data
   ```

2. **Frontend**: Create component in `frontend/src/components/`
   ```tsx
   export default function NewVisualizer({ data }: Props) {
     const vizData = useMemo(() => extractData(data), [data]);
     return <Plot data={...} />;
   }
   ```

3. **API**: Add endpoint in `backend/app/api/routes.py` if needed

### Modifying Model Architecture

**Location**: `backend/app/api/routes.py`

Change transformer_service initialization:
```python
transformer_service = TransformerService(
    d_model=512,  # Increase model size
    n_heads=8,
    n_encoder_layers=6,
    # ...
)
```

**Note**: Larger models = slower inference. Current size optimized for educational use.

### Adding New API Endpoint

1. Define Pydantic models (request/response)
2. Add route in `backend/app/api/routes.py`
3. Update `frontend/src/services/api.ts` with TypeScript interface
4. Use in frontend component

### Debugging Visualization Issues

1. **Check backend response**: Visit http://localhost:8000/docs, test endpoint
2. **Console log data**: `console.log(visualizationData)` in component
3. **Verify data shape**: Attention weights should be [batch, heads, seq, seq]
4. **Check Plotly config**: Ensure data format matches Plotly expectations

## Code Quality Standards

### Backend
- **Format**: Black (line length 100)
- **Lint**: Flake8
- **Types**: MyPy (strict mode)
- **Docstrings**: Google style, educational focus

### Frontend
- **Format**: Prettier (via ESLint)
- **Lint**: ESLint with TypeScript rules
- **Types**: Strict TypeScript
- **Components**: Functional components with hooks

## Educational Philosophy

**Every component should answer**:
- **What**: What does this do?
- **Why**: Why is it needed?
- **How**: How does it work mathematically?

**Example** (from attention.py):
```python
"""
Scaled Dot-Product Attention: The fundamental attention mechanism.

WHAT: Core attention calculation
WHY: Allows tokens to focus on relevant context
HOW: Q @ K^T scaled by sqrt(d_k), then weighted sum of V
"""
```

## Testing

### Backend Tests
- Location: `backend/tests/`
- Run: `pytest`
- Coverage: `pytest --cov=app`

### Frontend Tests
- Not yet implemented
- Recommended: Vitest + React Testing Library

## Deployment Considerations

### Backend
- Use Gunicorn/Uvicorn with multiple workers
- Consider GPU for larger models
- Implement caching for repeated queries

### Frontend
- Build: `npm run build` (outputs to `dist/`)
- Deploy to: Vercel, Netlify, or static hosting
- Set `VITE_API_URL` environment variable

## Performance Notes

**Backend**:
- Model init: ~500ms
- Inference (10 tokens): ~100-200ms
- Larger models significantly slower

**Frontend**:
- Use `useMemo` for expensive calculations
- Plotly responsive mode for window resizing
- Consider virtualization for large token lists

## Known Limitations

1. **Character-level tokenization**: Not suitable for production, just demo
2. **Small model**: Educational size, not for real tasks
3. **No training**: Only inference/visualization
4. **Limited languages**: ASCII characters only

## Future Enhancements

- Pre-trained model loading (GPT-2, BERT)
- Training visualization
- Transformer variants (BERT, GPT, T5)
- Multi-language support
- Export visualizations as images

## Troubleshooting

**Backend won't start**:
- Check Python version (3.9+)
- Activate virtual environment
- Install dependencies: `pip install -r requirements.txt`

**Frontend won't connect to backend**:
- Verify backend running on port 8000
- Check CORS settings in `backend/app/main.py`
- Check proxy in `frontend/vite.config.ts`

**Visualizations not showing**:
- Open browser console for errors
- Check data format in API response
- Verify Plotly.js is loaded

**Type errors**:
- Backend: Run `mypy app/` to check types
- Frontend: TypeScript strict mode catches issues at compile time

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Code walkthrough

## Project Status

**Current**: MVP complete with core visualizations
**Next**: Add more educational content, improve UX
**Long-term**: Support pre-trained models, add training visualization

---

When working on this codebase, prioritize **educational clarity** over optimization. The goal is to help people learn transformers, not to build a production system.
