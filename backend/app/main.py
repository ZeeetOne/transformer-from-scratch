"""
Transformer Interactive Visualization - FastAPI Application

Main application entry point for the backend API server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .api.routes import router

# Create FastAPI app
app = FastAPI(
    title="Transformer Interactive Visualization API",
    description="""
    Educational API for understanding Transformer architecture through interactive visualizations.

    This API provides:
    - Transformer inference with detailed intermediate outputs
    - Attention weight extraction for heatmap visualizations
    - Embedding progression tracking
    - Complete visualization data for all transformer components

    Built for educational purposes to help learners understand:
    - How attention mechanisms work
    - Token flow through encoder/decoder layers
    - Positional encoding patterns
    - Feed-forward transformations
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # Vite dev server (alternate port)
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with welcome message."""
    return {
        "message": "Transformer Interactive Visualization API",
        "version": "1.0.0",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "api_base": "/api/v1",
        "description": "Educational platform for learning transformer architecture"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error messages."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


# For development
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
