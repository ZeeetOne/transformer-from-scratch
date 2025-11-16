# Transformer Interactive Visualization

An educational platform for understanding transformer architecture through interactive, step-by-step visualizations. Built from scratch with detailed explanations at every stage.

![Architecture](https://img.shields.io/badge/Architecture-Transformer-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)
![React](https://img.shields.io/badge/React-18.2-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)

## 🎯 Project Vision

Help ML practitioners, students, and engineers understand transformers by visualizing:
- **Token embeddings** and positional encoding
- **Multi-head attention** mechanisms with heatmaps
- **Feed-forward networks** and layer transformations
- **Complete architecture** from input to output

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│         Frontend (React + TypeScript)           │
│  • Interactive visualizations (Plotly, D3.js)   │
│  • Attention heatmaps                           │
│  • Architecture diagrams                        │
└─────────────────────────────────────────────────┘
                       ↕ REST API
┌─────────────────────────────────────────────────┐
│         Backend (FastAPI + PyTorch)             │
│  • Transformer implementation from scratch      │
│  • Educational visualization data extraction    │
│  • Real-time inference                          │
└─────────────────────────────────────────────────┘
```

## 📚 Learning Objectives

After using this platform, you'll understand:

1. **Tokenization & Embeddings**: How text becomes vectors
2. **Positional Encoding**: Why sinusoidal patterns encode position
3. **Scaled Dot-Product Attention**: The core mechanism
4. **Multi-Head Attention**: Why multiple heads are better
5. **Encoder-Decoder Architecture**: How seq2seq works
6. **Complete Forward Pass**: End-to-end data flow

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- npm or yarn

### Backend Setup

```bash
# Navigate to backend
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

# Run the server
python -m app.main
# Or using uvicorn directly:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: http://localhost:8000
API documentation: http://localhost:8000/docs

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: http://localhost:3000

## 📖 Usage

1. **Enter Text**: Type or select example text in the control panel
2. **Run Inference**: Click "Run Inference" to process through transformer
3. **Explore Views**:
   - **Architecture**: See the complete transformer pipeline
   - **Embeddings**: Understand token and positional embeddings
   - **Attention**: Explore attention patterns across heads and layers
   - **Complete**: View everything at once

### Interactive Features

- **Layer Navigation**: Scroll through encoder/decoder layers
- **Head Selection**: Compare different attention heads
- **Attention Heatmaps**: See which tokens attend to which
- **Real-time Stats**: Entropy, focus patterns, activation sparsity

## 🔧 Project Structure

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
│   ├── tests/
│   └── requirements.txt
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── AttentionVisualizer.tsx
│   │   │   ├── EmbeddingVisualizer.tsx
│   │   │   ├── ArchitectureDiagram.tsx
│   │   │   └── ControlPanel.tsx
│   │   ├── services/
│   │   │   └── api.ts         # API client
│   │   ├── App.tsx
│   │   └── main.tsx
│   └── package.json
│
├── notebooks/                  # Jupyter notebooks (optional)
├── docs/                       # Documentation
├── PROJECT_PLAN.md            # Detailed project plan
├── CLAUDE.md                  # Development guide
└── README.md                  # This file
```

## 🎓 Educational Components

### 1. Token Embeddings
- Learn how tokens map to continuous vectors
- See embedding dimension visualization
- Understand scaling by √d_model

### 2. Positional Encoding
- Visualize sinusoidal patterns
- Compare different frequencies
- Understand relative position learning

### 3. Multi-Head Attention
- Interactive attention heatmaps
- Per-head entropy analysis
- Token-to-token flow visualization
- Compare attention patterns across heads

### 4. Feed-Forward Networks
- Activation heatmaps
- Dimension expansion/projection
- ReLU sparsity analysis

### 5. Architecture Overview
- Complete data flow diagram
- Layer-by-layer progression
- Encoder-decoder interaction

## 🛠️ Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests (if added)
cd frontend
npm test
```

### Code Quality

```bash
# Backend
black app/
flake8 app/
mypy app/

# Frontend
npm run lint
```

### Building for Production

```bash
# Backend (Docker)
docker build -t transformer-viz-backend ./backend

# Frontend
cd frontend
npm run build
```

## 📊 API Endpoints

### Main Endpoints

- `POST /api/v1/inference` - Run transformer inference
- `POST /api/v1/attention` - Get attention visualization
- `GET /api/v1/model/info` - Model architecture info
- `POST /api/v1/visualize/embeddings` - Embedding visualization
- `POST /api/v1/visualize/flow` - Attention flow data
- `POST /api/v1/visualize/complete` - Complete visualization data

Full API documentation: http://localhost:8000/docs

## 🎨 Technologies Used

### Backend
- **PyTorch**: Deep learning framework
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation
- **NumPy**: Numerical computations

### Frontend
- **React**: UI library
- **TypeScript**: Type-safe JavaScript
- **Plotly**: Interactive visualizations
- **Tailwind CSS**: Styling
- **Vite**: Build tool

## 🤝 Contributing

This is an educational project. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 Resources

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar's guide
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Vaswani et al. for "Attention is All You Need"
- The PyTorch and FastAPI communities
- All contributors and users

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

Built with ❤️ for education
