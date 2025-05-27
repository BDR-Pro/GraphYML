"""
Embedding service for GraphYML.
Provides a FastAPI server that generates embeddings using Hugging Face models.
"""
import os
import time
from typing import Dict, List, Any, Optional, Union

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Get environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
PORT = int(os.environ.get("PORT", 8000))

# Create FastAPI app
app = FastAPI(
    title="GraphYML Embedding Service",
    description="API for generating embeddings for GraphYML",
    version="1.0.0",
)

# Load model
print(f"Loading model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME}")


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    text: Union[str, List[str]]
    model: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embedding: List[Union[List[float], float]]
    model: str
    dimensions: int
    processing_time: float


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "GraphYML Embedding Service",
        "status": "running",
        "model": MODEL_NAME,
        "dimensions": model.get_sentence_embedding_dimension(),
    }


@app.get("/api/models")
async def get_models():
    """Get available models."""
    return {
        "models": [MODEL_NAME],
        "default": MODEL_NAME,
    }


@app.post("/api/embeddings", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    """
    Generate embeddings for text.
    
    Args:
        request: EmbeddingRequest with text to embed
        
    Returns:
        EmbeddingResponse with embedding vector
    """
    start_time = time.time()
    
    try:
        # Use specified model or default
        model_name = request.model or MODEL_NAME
        
        if model_name != MODEL_NAME:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} not available. Available models: {MODEL_NAME}",
            )
        
        # Generate embedding
        if isinstance(request.text, list):
            # Batch processing
            embeddings = model.encode(request.text)
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
        else:
            # Single text
            embedding = model.encode(request.text)
            # Convert to list
            embeddings_list = embedding.tolist()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return response
        return EmbeddingResponse(
            embedding=embeddings_list,
            model=MODEL_NAME,
            dimensions=model.get_sentence_embedding_dimension(),
            processing_time=processing_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embedding: {str(e)}",
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL_NAME}


if __name__ == "__main__":
    uvicorn.run("embedding_service:app", host="0.0.0.0", port=PORT, log_level="info")

