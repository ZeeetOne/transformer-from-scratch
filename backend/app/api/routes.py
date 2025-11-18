"""
API Routes for Transformer Visualization

Provides endpoints for:
- Model inference
- Visualization data extraction
- Model information
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from ..services.inference import TransformerService
from ..services.visualization import VisualizationExtractor
from ..services.gpt_service import GPTService

# Initialize router
router = APIRouter(prefix="/api/v1", tags=["transformer"])

# Initialize transformer service (singleton)
# In production, use dependency injection
transformer_service = TransformerService(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=256,  # Smaller for educational purposes
    n_heads=4,
    n_encoder_layers=2,
    n_decoder_layers=2,
    d_ff=1024,
    dropout=0.1
)

# Initialize GPT service for Mode 1 (Next Word Prediction)
# Check for trained checkpoint
import os

# Try to find trained model (in order of preference)
checkpoint_paths = [
    'checkpoints/best_model.pt',
    'checkpoints/final_model.pt',
]

checkpoint_path = None
for path in checkpoint_paths:
    if os.path.exists(path):
        checkpoint_path = path
        break

if checkpoint_path:
    print(f"Loading trained model from: {checkpoint_path}")
    gpt_service = GPTService(checkpoint_path=checkpoint_path)
else:
    print("No trained model found. Using untrained model.")
    print("To train a model, run: python train_gpt_model.py")
    gpt_service = GPTService(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_len=100,
        dropout=0.1
    )


# Request/Response Models
class InferenceRequest(BaseModel):
    """Request model for transformer inference."""
    source_text: str = Field(..., description="Source text to encode", min_length=1, max_length=200)
    target_text: Optional[str] = Field(None, description="Target text (optional, for teacher forcing)")
    generate: bool = Field(False, description="Whether to generate output autoregressively")
    max_gen_len: int = Field(50, description="Maximum generation length", ge=1, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "source_text": "Hello world",
                "target_text": None,
                "generate": True,
                "max_gen_len": 20
            }
        }


class InferenceResponse(BaseModel):
    """Response model for transformer inference."""
    source_text: str
    decoded_output: str
    source_tokens: List[int]
    target_tokens: List[int]
    mode: str
    visualization_data: Dict[str, Any]


class AttentionRequest(BaseModel):
    """Request model for attention visualization."""
    source_text: str = Field(..., description="Source text")
    layer_idx: int = Field(0, description="Layer index", ge=0)
    head_idx: Optional[int] = Field(None, description="Specific head index (optional)")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    architecture: str
    d_model: int
    n_encoder_layers: int
    n_decoder_layers: int
    vocab_size: int
    device: str
    total_parameters: int
    trainable_parameters: int


# API Endpoints

@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Transformer Interactive Visualization API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "inference": "/api/v1/inference",
            "attention": "/api/v1/attention",
            "model_info": "/api/v1/model/info"
        }
    }


@router.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """
    Run transformer inference and get visualization data.

    This endpoint:
    1. Tokenizes input text
    2. Runs forward pass through transformer
    3. Extracts comprehensive visualization data
    4. Returns results with decoded output

    Args:
        request: Inference request with source text and options

    Returns:
        Complete inference results with visualization data
    """
    try:
        result = transformer_service.run_inference(
            source_text=request.source_text,
            target_text=request.target_text,
            generate=request.generate,
            max_gen_len=request.max_gen_len
        )

        return InferenceResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/attention")
async def get_attention_visualization(request: AttentionRequest):
    """
    Get attention visualization data for a specific layer/head.

    This endpoint extracts and formats attention weights for heatmap visualization.

    Args:
        request: Request specifying text and layer/head indices

    Returns:
        Formatted attention data ready for visualization
    """
    try:
        # Run inference to get visualization data
        result = transformer_service.run_inference(
            source_text=request.source_text,
            target_text=None,
            generate=False
        )

        viz_data = result['visualization_data']

        # Extract attention heatmaps
        attention_data = VisualizationExtractor.extract_attention_heatmaps(
            viz_data,
            layer_idx=request.layer_idx
        )

        # If specific head requested, filter
        if request.head_idx is not None:
            heatmaps = attention_data.get('heatmaps', [])
            if request.head_idx < len(heatmaps):
                attention_data['heatmaps'] = [heatmaps[request.head_idx]]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Head {request.head_idx} not found. Available: 0-{len(heatmaps)-1}"
                )

        # Add token information
        attention_data['tokens'] = {
            'source': result['source_tokens'],
            'text': request.source_text
        }

        return attention_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attention extraction failed: {str(e)}")


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get model architecture information.

    Returns:
        Model specifications including layer counts, dimensions, parameters
    """
    try:
        info = transformer_service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/visualize/embeddings")
async def get_embedding_visualization(source_text: str):
    """
    Get embedding progression visualization.

    Shows how embeddings transform: Token → +Position → Layers

    Args:
        source_text: Input text

    Returns:
        Embedding progression data
    """
    try:
        result = transformer_service.run_inference(
            source_text=source_text,
            generate=False
        )

        viz_data = result['visualization_data']

        embedding_data = VisualizationExtractor.extract_embedding_progression(viz_data)
        positional_data = VisualizationExtractor.extract_positional_encoding_patterns(viz_data)

        return {
            "tokens": result['source_tokens'],
            "text": source_text,
            "embedding_progression": embedding_data,
            "positional_encoding": positional_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding visualization failed: {str(e)}")


@router.post("/visualize/flow")
async def get_attention_flow(
    source_text: str,
    layer_idx: int = 0,
    head_idx: int = 0
):
    """
    Get attention flow visualization showing token-to-token connections.

    Args:
        source_text: Input text
        layer_idx: Layer index
        head_idx: Attention head index

    Returns:
        Attention flow data with top-k connections for each token
    """
    try:
        result = transformer_service.run_inference(
            source_text=source_text,
            generate=False
        )

        viz_data = result['visualization_data']

        flow_data = VisualizationExtractor.extract_attention_flow(
            viz_data,
            layer_idx=layer_idx,
            head_idx=head_idx
        )

        flow_data['tokens'] = result['source_tokens']
        flow_data['text'] = source_text

        return flow_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flow visualization failed: {str(e)}")


@router.post("/visualize/complete")
async def get_complete_visualization(request: InferenceRequest):
    """
    Get complete visualization data formatted for frontend.

    This is a convenience endpoint that returns all visualization data
    in a frontend-friendly format.

    Args:
        request: Inference request

    Returns:
        Complete formatted visualization data
    """
    try:
        result = transformer_service.run_inference(
            source_text=request.source_text,
            target_text=request.target_text,
            generate=request.generate,
            max_gen_len=request.max_gen_len
        )

        viz_data = result['visualization_data']

        # Format for frontend
        formatted_data = VisualizationExtractor.format_for_frontend(
            viz_data=viz_data,
            source_tokens=[str(t) for t in result['source_tokens']],
            target_tokens=[str(t) for t in result.get('target_tokens', [])]
        )

        # Add inference results
        formatted_data['inference'] = {
            'source_text': result['source_text'],
            'decoded_output': result['decoded_output'],
            'mode': result['mode']
        }

        return formatted_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Complete visualization failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "transformer-visualization",
        "model_loaded": True
    }


# Mode 1: Next Word Prediction Endpoints

class NextWordRequest(BaseModel):
    """Request model for next word prediction (Mode 1)."""
    input_text: str = Field(..., description="Input text context", min_length=1, max_length=200)

    class Config:
        json_schema_extra = {
            "example": {
                "input_text": "I eat"
            }
        }


@router.post("/predict-next-word")
async def predict_next_word(request: NextWordRequest):
    """
    Predict the next word/token based on input context (Mode 1: GPT-style).

    This endpoint:
    1. Tokenizes input text
    2. Runs forward pass through GPT-style decoder
    3. Predicts next token with probabilities
    4. Returns step-by-step visualization data

    Args:
        request: Request with input text

    Returns:
        Prediction results with detailed visualization data for each step
    """
    try:
        result = gpt_service.predict_next_word(request.input_text)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Next word prediction failed: {str(e)}"
        )
