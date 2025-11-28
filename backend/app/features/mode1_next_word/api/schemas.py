"""
Pydantic schemas for Mode 1 API.
"""

from pydantic import BaseModel, Field


class NextWordRequest(BaseModel):
    """Request model for next word prediction (Mode 1)."""
    input_text: str = Field(..., description="Input text context", min_length=1, max_length=200)

    class Config:
        json_schema_extra = {
            "example": {
                "input_text": "I eat"
            }
        }


class NextWordResponse(BaseModel):
    """Response model for next word prediction."""
    input_text: str
    predicted_token: str
    predicted_word: str
    confidence: float
    top_predictions: list
    steps: dict
    raw_visualization: dict
