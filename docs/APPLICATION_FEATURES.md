# Application Features

This document describes the core features and capabilities of the Transformer Interactive Visualization platform.

## Overview

The platform provides an educational interface for understanding how transformers work through interactive, step-by-step visualizations. Currently, the application supports one main mode with plans for expansion.

## Current Modes

### Mode 1: Next Word Prediction (Mini-GPT)

**Purpose**: Demonstrates how a decoder-only transformer predicts the next word in a sequence, similar to GPT models.

**Use Case**: Understanding autoregressive language models and text generation.

#### How It Works

1. **Input**: User enters a text sequence (e.g., "I eat")
2. **Processing**: The transformer processes the text through multiple stages
3. **Output**: Predicted next word with confidence scores
4. **Visualization**: Each processing step is visualized interactively

#### Processing Pipeline

The Mode 1 feature breaks down transformer inference into 6 educational steps:

##### Step 1: Tokenization
- **What**: Converts input text into discrete tokens
- **Current Implementation**: Character-level tokenization
- **Special Tokens**: `<SOS>` (Start of Sequence), `<EOS>` (End of Sequence), `<PAD>` (Padding), `<UNK>` (Unknown)
- **Visualization**: Shows each token with its ID
- **Educational Value**: Understand how text becomes numerical data

##### Step 2: Embedding + Positional Encoding
- **What**: Converts tokens to dense vectors and adds position information
- **Components**:
  - **Word Embeddings (WE)**: Learned semantic meaning of each token
  - **Positional Encoding (PE)**: Sinusoidal patterns encoding position
  - **Final Embedding**: WE + PE (element-wise addition)
- **Visualization**:
  - Color-coded grids showing embedding values
  - Sinusoidal wave patterns for positional encoding
  - Side-by-side comparison showing how position changes meaning
- **Educational Value**: Understand how transformers handle word meaning and position

##### Step 3: Self-Attention & Multi-Head Attention
- **What**: Allows each token to attend to all previous tokens (causal attention)
- **Components**:
  - **Query (Q)**: What am I looking for?
  - **Key (K)**: What do I contain?
  - **Value (V)**: What information do I have?
  - **Attention Formula**: `Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V`
- **Multi-Head**: Multiple parallel attention mechanisms (default: 4 heads)
- **Visualization**:
  - Three-panel layout showing mechanism flow
  - Panel A: Input → Q/K/V projections
  - Panel B: Scaled dot-product attention calculation
  - Panel C: Multi-head attention + attention pattern graph
  - Interactive head selection
  - Attention lines showing token-to-token connections
- **Educational Value**: Understand the "magic" of attention and why multiple heads matter

##### Step 4: Feedforward Network
- **What**: Position-wise fully connected neural network
- **Architecture**: Two linear transformations with ReLU activation
- **Dimension**: Expands to hidden dimension (default: 1024), then projects back to model dimension
- **Visualization**:
  - Activation heatmaps showing neuron activations
  - Dimension expansion/contraction flow
  - ReLU sparsity analysis
- **Educational Value**: Understand how transformers process information at each position

##### Step 5: Output Layer (Softmax)
- **What**: Converts final hidden states to probability distribution over vocabulary
- **Process**:
  - Linear projection to vocabulary size
  - Softmax normalization
- **Visualization**:
  - Top predictions with probability bars
  - Distribution heatmap
  - Confidence scores
- **Educational Value**: Understand how transformers make predictions

##### Step 6: Prediction Result
- **What**: Final predicted word with comprehensive statistics
- **Displays**:
  - Predicted token and word
  - Confidence score
  - Top-k predictions with probabilities
  - Alternative predictions
- **Educational Value**: Interpret model outputs and confidence

## Model Architecture

### Configuration (Educational Size)

```yaml
Model Dimension (d_model): 256
Number of Attention Heads: 4
Number of Encoder Layers: 2
Number of Decoder Layers: 2
Feedforward Dimension (d_ff): 1024
Dropout: 0.1
Max Sequence Length: 100
Vocabulary Size: Character-based (ASCII)
```

**Note**: The model uses an educational size optimized for visualization and learning, not for production-level performance.

### Attention Mechanism

- **Type**: Causal (masked) self-attention
- **Masking**: Prevents attending to future tokens
- **Scaling**: Attention scores scaled by √d_k to prevent softmax saturation
- **Heads**: Each head learns different attention patterns

### Transformer Components

All components are implemented from scratch in PyTorch:

1. **Token Embeddings**: Learned embedding layer
2. **Positional Encoding**: Fixed sinusoidal encoding
3. **Multi-Head Attention**: Parallel attention mechanisms
4. **Feedforward Networks**: Position-wise MLP
5. **Layer Normalization**: Pre-normalization (before attention/FF)
6. **Residual Connections**: Skip connections around each sublayer

## API Integration

### Backend Endpoints

#### POST `/api/v1/predict/next-word`
Predicts the next word for Mode 1.

**Request**:
```json
{
  "text": "I eat"
}
```

**Response**:
```json
{
  "input_text": "I eat",
  "predicted_token": "s",
  "predicted_word": "s",
  "confidence": 0.234,
  "top_predictions": [
    {"token": "s", "probability": 0.234},
    {"token": "a", "probability": 0.189},
    ...
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

#### GET `/api/v1/model/info`
Returns model architecture information.

**Response**:
```json
{
  "d_model": 256,
  "n_heads": 4,
  "n_encoder_layers": 2,
  "n_decoder_layers": 2,
  "d_ff": 1024,
  "vocab_size": 128
}
```

## Educational Philosophy

### Design Principles

1. **Clarity Over Optimization**: Code is written for understanding, not performance
2. **Step-by-Step**: Complex processes broken into digestible steps
3. **Visual Learning**: Every concept has a visual representation
4. **Interactive Exploration**: Users can interact with visualizations
5. **Mathematical Grounding**: Formulas and explanations provided

### Learning Path

The application is designed to be explored in order:

1. **Start with Mode 1**: Understand basic transformer operation
2. **Explore Each Step**: Click through the 6 processing steps
3. **Interact with Visualizations**: Click heads, hover elements, toggle comparisons
4. **Experiment with Input**: Try different text inputs
5. **Compare Patterns**: Observe how attention patterns change

## Limitations

### Current Limitations

1. **Character-Level Tokenization**: Not suitable for real-world use, only for demonstration
2. **Small Model**: Educational size, not capable of sophisticated language understanding
3. **Limited Vocabulary**: ASCII characters only
4. **No Training Interface**: Only inference/visualization, no training capability
5. **Single Mode**: Only next-word prediction currently available

### Performance Considerations

- **Inference Time**: ~100-200ms for 10 tokens (CPU)
- **Model Size**: ~5MB (small, optimized for web)
- **Browser Performance**: Visualizations may be slow on older devices
- **Sequence Length**: Limited to 100 tokens for performance

## Future Modes (Planned)

### Mode 2: Translation (Seq2Seq)
- Full encoder-decoder architecture
- Visualize cross-attention between encoder and decoder
- Interactive language pair selection

### Mode 3: Masked Language Modeling (BERT-style)
- Bidirectional attention
- Mask prediction
- Understanding contextualized embeddings

### Mode 4: Custom Model Loading
- Load pre-trained models (GPT-2, BERT, etc.)
- Visualize real model behavior
- Compare different architectures

## Technical Details

### Data Flow

```
User Input
    ↓
Frontend (React)
    ↓ REST API
Backend (FastAPI)
    ↓
Transformer Model (PyTorch)
    ↓ Forward Pass
Extract Visualization Data
    ↓ JSON Response
Frontend Visualizations
    ↓
Interactive Display
```

### Visualization Data Extraction

Each layer in the transformer returns both:
- **Output**: Tensor for next layer
- **Visualization Data**: Dictionary with intermediate values

This design pattern enables detailed visualization without modifying model architecture.

### State Management

- **Local State**: React hooks (useState, useMemo)
- **No Global State**: Simple component-level state
- **API Caching**: None currently (each request is fresh)

## Resources

### Related Papers
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) - GPT-2

### Educational Resources
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

## Contributing

To add new features or modes:

1. **Backend**: Implement model logic in `backend/app/models/`
2. **API**: Add endpoint in `backend/app/api/routes.py`
3. **Frontend**: Create page in `frontend/src/pages/`
4. **Visualizations**: Add components in `frontend/src/components/`
5. **Documentation**: Update this file with new features

---

*Last Updated: 2025-11-27*
