# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Transformer Interactive Visualization** - An educational platform for learning transformer architecture through interactive visualizations. Built from scratch with detailed explanations at every stage.

**Purpose**: Help ML practitioners, students, and engineers understand transformers by visualizing embeddings, attention mechanisms, and complete architecture through step-by-step interactive demonstrations.

**Architecture**: Full-stack application with Python/FastAPI backend and React/TypeScript frontend.

**Current Status**: Mode 1 (Next Word Prediction) fully implemented with comprehensive educational visualizations.

## Documentation

Comprehensive documentation is organized in the `/docs` folder:

- **[docs/APPLICATION_FEATURES.md](docs/APPLICATION_FEATURES.md)** - Complete feature documentation, API endpoints, model architecture, and Mode 1 pipeline details
- **[docs/WEBSITE_FEATURES.md](docs/WEBSITE_FEATURES.md)** - UI/UX documentation, visualization components, design system, and interaction patterns
- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - Original project vision and technical decisions
- **[README.md](README.md)** - User-facing documentation and quick start guide

**When developing**: Always consult the relevant documentation above for detailed information about features, APIs, and UI components.

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
- **Visualizations**: Plotly.js, D3.js, SVG
- **State**: React hooks (useState, useMemo)
- **Routing**: React Router DOM

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
├── pages/
│   ├── Home.tsx                    # Landing page
│   ├── Applications.tsx            # Mode selection hub
│   └── Mode1.tsx                   # Next word prediction mode
├── components/
│   └── mode1/                      # Mode 1 visualizers
│       ├── TokenizationVisualizer.tsx
│       ├── EmbeddingVisualizerV2.tsx       # Step 2 (V2: Educational redesign)
│       ├── AttentionVisualizerV2.tsx       # Step 3 (V2: Three-panel layout)
│       ├── FeedforwardVisualizer.tsx
│       ├── SoftmaxVisualizer.tsx
│       └── PredictionVisualizer.tsx
├── services/
│   └── api.ts                      # Backend API client
├── App.tsx                         # Main app with routing
└── main.tsx                        # Entry point
```

**Key Patterns**:
- Pages use React Router for navigation
- Components receive prediction result data as props
- Visualization data extracted using `useMemo` for performance
- V2 visualizers implement textbook-quality educational designs

## Current Features (Mode 1)

### Mode 1: Next Word Prediction (Mini-GPT)

**Path**: `/applications/mode1`

**6-Step Pipeline Visualization**:

1. **Tokenization** - Character-level tokenization with special tokens
2. **Embedding + Positional Encoding** - Word embeddings + sinusoidal PE visualization
   - Component: `EmbeddingVisualizerV2.tsx`
   - Features: Grid visualization, sinusoidal waves, position comparison
3. **Self-Attention & Multi-Head Attention** - Three-panel integrated layout
   - Component: `AttentionVisualizerV2.tsx`
   - Features: Q/K/V projections, attention calculation, multi-head graph, interactive head selection
4. **Feedforward Network** - Dimension expansion and activation visualization
5. **Output Layer (Softmax)** - Probability distribution over vocabulary
6. **Prediction Result** - Final predicted word with confidence scores

See [docs/APPLICATION_FEATURES.md](docs/APPLICATION_FEATURES.md) for complete Mode 1 documentation.

## Important Implementation Details

### Transformer Model

**Configuration** (educational size):
- d_model: 256 (model dimension)
- n_heads: 4 (attention heads)
- n_encoder_layers: 2
- n_decoder_layers: 2
- d_ff: 1024 (feed-forward dimension)
- max_seq_length: 100
- dropout: 0.1

**Location**: Backend model configured in `backend/app/api/routes.py` (transformer_service initialization)

**Scaling**: All embeddings scaled by √d_model (Attention is All You Need paper)

### Attention Mechanism

**Location**: `backend/app/models/attention.py`

**Critical Components**:
1. `ScaledDotProductAttention`: Core attention (Q, K, V)
2. `MultiHeadAttention`: Parallel heads with projections

**Formula**: `Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V`

**Causal Masking**: For autoregressive prediction (prevents attending to future tokens)

**Visualization Data**: Each head returns attention_weights array for heatmap rendering

### API Endpoints

**Current Endpoints** (`/api/v1/...`):
- `POST /predict/next-word` - Run Mode 1 prediction, get complete visualization data
- `GET /model/info` - Model architecture specifications

**Endpoint Details**:

#### POST `/api/v1/predict/next-word`
```json
Request: {
  "text": "I eat"
}

Response: {
  "input_text": "I eat",
  "predicted_token": "s",
  "predicted_word": "s",
  "confidence": 0.234,
  "top_predictions": [...],
  "steps": {
    "tokenization": {...},
    "embeddings": {...},
    "attention": {...},
    "feedforward": {...},
    "output": {...}
  }
}
```

See [docs/APPLICATION_FEATURES.md](docs/APPLICATION_FEATURES.md) for complete API documentation.

### Tokenization

**Current**: Simple character-level tokenizer (demo purposes)
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- Location: `backend/app/services/inference.py`
- Vocabulary: ASCII characters (128 chars)

**Future**: Can swap for tiktoken, sentencepiece, or HuggingFace tokenizer

### Visualization Data Flow

1. User enters text in Mode1 page input
2. Frontend calls `predictNextWord()` API function
3. Backend runs forward pass, extracts viz data at each layer
4. Frontend receives complete prediction result
5. User navigates through 6 steps
6. Each visualizer component renders interactive visualization

## Advanced Visualizer Components

### EmbeddingVisualizerV2 (Step 2)

**Design**: Educational three-section layout showing:
- Word Embeddings (WE): Learned semantic vectors as colored grids
- Positional Encoding (PE): Sinusoidal patterns with wave visualization
- Final Embedding: WE + PE with comparison feature

**Interactive Features**:
- Hover tooltips showing exact values
- "Show How Position Changes Meaning" toggle
- Color-coded grids (8 dimensions shown)
- Animated transitions

**Educational Value**: Students understand how transformers handle meaning AND position.

### AttentionVisualizerV2 (Step 3)

**Design**: Three-panel integrated layout:
- **Panel A**: Input → Q/K/V projections (mechanism)
- **Panel B**: Scaled dot-product attention calculation (step-by-step)
- **Panel C**: Multi-head attention + attention pattern graph (behavior)

**Interactive Features**:
- Head selector buttons (4 heads, color-coded)
- Attention graph with token-to-token lines
- Line thickness = attention weight
- Hover details on Q/K/V vectors
- Animated line drawing

**Educational Value**: Shows both HOW attention works (mechanism) and WHAT it produces (patterns).

See [docs/WEBSITE_FEATURES.md](docs/WEBSITE_FEATURES.md) for complete UI component documentation.

## Common Development Tasks

### Adding a New Visualization to Mode 1

1. **Backend**: Ensure data is extracted in prediction response
2. **Frontend**: Create visualizer component in `frontend/src/components/mode1/`
3. **Integration**: Add to Mode1.tsx steps array
4. **Styling**: Use consistent color palette and design system (see docs/WEBSITE_FEATURES.md)

Example:
```tsx
// frontend/src/components/mode1/NewVisualizer.tsx
export default function NewVisualizer({ data }: Props) {
  const vizData = useMemo(() => processData(data), [data]);

  return (
    <div className="space-y-4">
      {/* Visualization content */}
    </div>
  );
}
```

### Adding a New Mode (Mode 2, 3, etc.)

1. **Backend**:
   - Add model/service logic in `backend/app/services/`
   - Create API endpoint in `backend/app/api/routes.py`
2. **Frontend**:
   - Create page: `frontend/src/pages/Mode2.tsx`
   - Create visualizer components: `frontend/src/components/mode2/`
   - Add route in `App.tsx`
   - Add to Applications page
3. **Documentation**: Update docs/APPLICATION_FEATURES.md

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

### Updating Visualizations

**Color Palette** (maintain consistency):
```javascript
// Token colors (consistent across all visualizations)
const tokenColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', ...];

// Attention head colors
const headColors = ['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B', ...];

// Component colors
Q: '#3B82F6'  // Blue
K: '#10B981'  // Green
V: '#F59E0B'  // Yellow
```

See [docs/WEBSITE_FEATURES.md](docs/WEBSITE_FEATURES.md) for complete design system.

### Debugging Visualization Issues

1. **Check backend response**: Visit http://localhost:8000/docs, test endpoint directly
2. **Console log data**: `console.log(result)` in Mode1.tsx after API call
3. **Verify data shape**: Check that data matches visualizer expectations
4. **Component props**: Ensure correct data passed to visualizer components
5. **Browser console**: Look for React errors or warnings

## Code Quality Standards

### Backend
- **Format**: Black (line length 100)
- **Lint**: Flake8
- **Types**: MyPy (strict mode)
- **Docstrings**: Google style with educational focus (What/Why/How)

### Frontend
- **Format**: Prettier (via ESLint)
- **Lint**: ESLint with TypeScript rules
- **Types**: Strict TypeScript
- **Components**: Functional components with hooks
- **Naming**: PascalCase for components, camelCase for functions

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

Formula: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
"""
```

**Visualization Design Principles**:
1. **Clarity Over Optimization** - Readable code, clear visuals
2. **Step-by-Step** - Complex processes broken into digestible steps
3. **Interactive Learning** - Users can explore and experiment
4. **Mathematical Grounding** - Formulas and explanations provided
5. **Consistent Design** - Color palette and patterns maintained throughout

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
- Environment variables for model configuration

### Frontend
- Build: `npm run build` (outputs to `dist/`)
- Deploy to: Vercel, Netlify, or static hosting
- Set `VITE_API_URL` environment variable for backend URL
- Enable compression and caching

## Performance Notes

**Backend**:
- Model initialization: ~500ms
- Inference (10 tokens): ~100-200ms (CPU)
- Larger models significantly slower
- Consider model caching for repeated use

**Frontend**:
- Use `useMemo` for expensive calculations (especially in visualizers)
- Lazy load mode pages with React.lazy()
- SVG rendering for attention graphs (scalable, performant)
- Debounce input validation
- Optimize re-renders with React.memo where appropriate

## Known Limitations

1. **Character-level tokenization**: Not suitable for production, demo only
2. **Small model**: Educational size, not for real language tasks
3. **No training interface**: Only inference/visualization
4. **Limited vocabulary**: ASCII characters only (128 chars)
5. **Single language**: English-focused examples
6. **No model persistence**: Model reinitialized on server restart

## Future Enhancements

**Planned Modes**:
- Mode 2: Translation (Seq2Seq) - Full encoder-decoder
- Mode 3: Masked Language Modeling (BERT-style)
- Mode 4: Custom Model Loading (GPT-2, BERT)

**UI Enhancements**:
- Dark mode toggle
- Export visualizations as PNG/SVG
- Animation playback controls
- Comparison mode (side-by-side inputs)
- Shareable configuration links
- Tutorial tooltips and onboarding

**Technical Enhancements**:
- Pre-trained model loading
- Training visualization
- Multi-language tokenization
- WebSocket for real-time updates
- Model fine-tuning interface

See [docs/APPLICATION_FEATURES.md](docs/APPLICATION_FEATURES.md) for complete roadmap.

## Troubleshooting

**Backend won't start**:
- Check Python version (3.9+): `python --version`
- Activate virtual environment
- Install dependencies: `pip install -r requirements.txt`
- Check port 8000 not in use

**Frontend won't connect to backend**:
- Verify backend running: http://localhost:8000/docs
- Check CORS settings in `backend/app/main.py`
- Check API URL in `frontend/src/services/api.ts`
- Verify network connectivity

**Visualizations not showing**:
- Open browser console (F12) for errors
- Check API response in Network tab
- Verify data format matches component expectations
- Check component props passed correctly

**Type errors**:
- Backend: Run `mypy app/` to check types
- Frontend: Check TypeScript errors in terminal (`npm run dev`)
- Ensure interfaces match between API and frontend

**Performance issues**:
- Reduce model size in routes.py
- Enable React strict mode to find issues
- Profile with React DevTools
- Check for unnecessary re-renders

## Project File Structure

```
transformer-from-scratch-draft/
├── backend/                    # Python/FastAPI backend
│   ├── app/
│   │   ├── models/            # Transformer implementation
│   │   ├── services/          # Business logic
│   │   ├── api/               # API routes
│   │   └── main.py           # FastAPI app
│   ├── tests/                # Backend tests
│   └── requirements.txt      # Python dependencies
│
├── frontend/                   # React/TypeScript frontend
│   ├── src/
│   │   ├── pages/            # Page components (Home, Applications, Mode1)
│   │   ├── components/       # Reusable components
│   │   │   └── mode1/       # Mode 1 visualizers
│   │   ├── services/         # API client
│   │   ├── App.tsx          # Main app with routing
│   │   └── main.tsx         # Entry point
│   └── package.json         # Node dependencies
│
├── docs/                       # Documentation
│   ├── APPLICATION_FEATURES.md # Feature & API docs
│   └── WEBSITE_FEATURES.md    # UI/UX docs
│
├── CLAUDE.md                   # This file (developer guide)
├── PROJECT_PLAN.md            # Project vision & plan
└── README.md                  # User-facing docs
```

## Quick Reference

**Start Development**:
```bash
# Terminal 1 (Backend)
cd backend && python -m app.main

# Terminal 2 (Frontend)
cd frontend && npm run dev
```

**Access Application**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Key Files to Know**:
- Backend API: `backend/app/api/routes.py`
- Mode 1 Page: `frontend/src/pages/Mode1.tsx`
- Attention Viz: `frontend/src/components/mode1/AttentionVisualizerV2.tsx`
- Embedding Viz: `frontend/src/components/mode1/EmbeddingVisualizerV2.tsx`

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide by Jay Alammar
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP code walkthrough
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) - Interactive explainer

## Project Status

**Current**: ✅ Mode 1 (Next Word Prediction) complete with advanced educational visualizations
**Next**: 🔜 Mode 2 (Translation), UI enhancements, dark mode
**Long-term**: 🎯 Pre-trained model support, training visualization, multiple transformer variants

---

**When working on this codebase, prioritize educational clarity over optimization.**

The goal is to help people learn transformers through clear code, comprehensive documentation, and beautiful interactive visualizations.

For detailed information, always refer to:
- Feature details → [docs/APPLICATION_FEATURES.md](docs/APPLICATION_FEATURES.md)
- UI/UX details → [docs/WEBSITE_FEATURES.md](docs/WEBSITE_FEATURES.md)
