"""
API Router for Mode 1: Next Word Prediction
"""

from fastapi import APIRouter, HTTPException
import os
from pathlib import Path

from .schemas import NextWordRequest, NextWordResponse
from ..services.prediction_service import PredictionService


# Initialize router
router = APIRouter(prefix="/api/v1", tags=["mode1-next-word"])

# Initialize prediction service (singleton)
# Check for trained checkpoint in mode1_next_word/checkpoints/
feature_dir = Path(__file__).parent.parent
checkpoint_paths = [
    feature_dir / 'checkpoints' / 'best_model.pt',
    feature_dir / 'checkpoints' / 'final_model.pt',
]

checkpoint_path = None
for path in checkpoint_paths:
    if path.exists():
        checkpoint_path = str(path)
        break

if checkpoint_path:
    print(f"[Mode 1] Loading trained model from: {checkpoint_path}")
    prediction_service = PredictionService(checkpoint_path=checkpoint_path)
else:
    print("[Mode 1] No trained model found. Using untrained model.")
    print("To train a model, run: python -m app.features.mode1_next_word.train")
    prediction_service = PredictionService(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_len=100,
        dropout=0.1
    )


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
        result = prediction_service.predict_next_word(request.input_text)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Next word prediction failed: {str(e)}"
        )
