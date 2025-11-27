# Transformer Interactive Visualization

An educational platform for understanding transformer architecture through interactive, step-by-step visualizations. Built from scratch with detailed explanations at every stage.

![Architecture](https://img.shields.io/badge/Architecture-Transformer-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)
![React](https://img.shields.io/badge/React-18.2-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)

## Overview

This project helps ML practitioners, students, and engineers understand transformers by visualizing:
- **Token embeddings** and positional encoding
- **Multi-head attention** mechanisms with interactive heatmaps
- **Feed-forward networks** and layer transformations
- **Complete architecture** from input to output

### Key Features

- **Mode 1: Next Word Prediction (Mini-GPT)** - Autoregressive language model with step-by-step visualization
- **Interactive Visualizations** - Explore attention patterns, embeddings, and more
- **Educational Focus** - Clear explanations and mathematical foundations
- **Built from Scratch** - PyTorch transformer implementation with detailed comments

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- npm or yarn

### Installation & Running

#### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m app.main
```

**Backend URL**: http://localhost:8000
**API Documentation**: http://localhost:8000/docs

#### 2. Frontend Setup

Open a new terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Frontend URL**: http://localhost:3000

### First Steps

1. Open http://localhost:3000 in your browser
2. Navigate to **Applications** → **Mode 1: Next Word Prediction**
3. Enter text (e.g., "I eat")
4. Click **"Predict Next Word"**
5. Explore the 6 visualization steps:
   - Step 1: Tokenization
   - Step 2: Embeddings + Positional Encoding
   - Step 3: Self-Attention & Multi-Head Attention
   - Step 4: Feedforward Network
   - Step 5: Output Layer (Softmax)
   - Step 6: Prediction Result

## Documentation

Comprehensive documentation is available in the `/docs` folder:

- **[Application Features](docs/APPLICATION_FEATURES.md)** - Detailed feature documentation, API endpoints, model architecture, and processing pipeline
- **[Website Features](docs/WEBSITE_FEATURES.md)** - UI components, visual design system, interaction patterns, and accessibility
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for working with this codebase
- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - Original project vision and technical decisions

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Frontend (React + TypeScript)           │
│  • Interactive visualizations (Plotly, D3.js)   │
│  • Step-by-step exploration                     │
│  • Responsive, accessible UI                    │
└─────────────────────────────────────────────────┘
                       ↕ REST API
┌─────────────────────────────────────────────────┐
│         Backend (FastAPI + PyTorch)             │
│  • Transformer implementation from scratch      │
│  • Educational visualization data extraction    │
│  • Real-time inference                          │
└─────────────────────────────────────────────────┘
```

### Technology Stack

**Backend**:
- Python 3.9+ (PyTorch, FastAPI, NumPy)
- Transformer implementation from scratch
- RESTful API with automatic documentation

**Frontend**:
- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- Plotly.js & D3.js (visualizations)

## Project Structure

```
transformer-visualization/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── models/            # Transformer implementation
│   │   │   ├── embeddings.py  # Token & positional embeddings
│   │   │   ├── attention.py   # Multi-head attention
│   │   │   ├── layers.py      # Encoder/decoder layers
│   │   │   └── transformer.py # Complete model
│   │   ├── services/
│   │   │   ├── inference.py   # Model inference service
│   │   │   └── visualization.py # Data extraction
│   │   ├── api/
│   │   │   └── routes.py      # API endpoints
│   │   └── main.py            # FastAPI app
│   └── requirements.txt
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   │   └── mode1/         # Mode 1 visualizers
│   │   ├── pages/             # Page components
│   │   ├── services/
│   │   │   └── api.ts         # API client
│   │   └── App.tsx
│   └── package.json
│
├── docs/                       # Documentation
│   ├── APPLICATION_FEATURES.md # Feature documentation
│   └── WEBSITE_FEATURES.md    # UI/UX documentation
│
├── CLAUDE.md                   # Development guide
├── PROJECT_PLAN.md            # Project vision
└── README.md                  # This file
```

## Educational Visualizations

### Mode 1: Next Word Prediction

Interactive 6-step pipeline visualization:

1. **Tokenization** - See how text becomes tokens
2. **Embeddings + Positional Encoding** - Understand semantic meaning + position
   - Word embedding grids
   - Sinusoidal positional patterns
   - Position comparison feature
3. **Self-Attention & Multi-Head Attention** - The "magic" of transformers
   - Three-panel integrated layout
   - Q/K/V projection visualization
   - Attention pattern graphs with interactive head selection
4. **Feedforward Network** - Position-wise transformations
5. **Softmax Output** - Probability distribution over vocabulary
6. **Prediction Result** - Final prediction with confidence scores

Each step includes:
- Visual representations
- Mathematical formulas
- Interactive elements
- Educational annotations

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Code quality
black app/              # Format
flake8 app/            # Lint
mypy app/              # Type checking
```

### Building for Production

```bash
# Backend (Docker)
docker build -t transformer-viz-backend ./backend

# Frontend
cd frontend
npm run build
```

## API Endpoints

- `POST /api/v1/predict/next-word` - Run next-word prediction (Mode 1)
- `GET /api/v1/model/info` - Get model architecture information

Full API documentation: http://localhost:8000/docs

## Learning Resources

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide by Jay Alammar
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP code walkthrough

## Future Plans

### Upcoming Modes

- **Mode 2**: Translation (Seq2Seq) - Full encoder-decoder architecture
- **Mode 3**: Masked Language Modeling (BERT-style) - Bidirectional attention
- **Mode 4**: Custom Model Loading - Load and visualize pre-trained models

### UI Enhancements

- Dark mode
- Export visualizations (PNG/SVG)
- Animation playback controls
- Comparison mode (side-by-side inputs)
- Shareable configuration links

## Contributing

Contributions are welcome! This is an educational project focused on clarity and learning.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Vaswani et al. for "Attention is All You Need"
- The PyTorch and FastAPI communities
- Jay Alammar for "The Illustrated Transformer"
- All contributors and users

## Contact

For questions, feedback, or issues, please open an issue on GitHub.

---

**Built with educational clarity in mind** - Helping people understand transformers through interactive visualization.

*For detailed feature documentation, see [docs/APPLICATION_FEATURES.md](docs/APPLICATION_FEATURES.md)*
*For UI/UX documentation, see [docs/WEBSITE_FEATURES.md](docs/WEBSITE_FEATURES.md)*
