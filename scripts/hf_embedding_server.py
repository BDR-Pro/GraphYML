#!/usr/bin/env python3
"""
Simple API server for Hugging Face sentence-transformers embeddings.
Provides a compatible API for GraphYML to use with Hugging Face models.

Usage:
    python hf_embedding_server.py --model all-MiniLM-L6-v2 --port 8000
"""
import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper for sentence-transformers model."""
    
    def __init__(self, model_name):
        """Initialize with model name."""
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Model loaded: {model_name}")
    
    def get_embedding(self, text):
        """Generate embedding for text."""
        if not text:
            return None
        
        # Generate embedding
        embedding = self.model.encode(text)
        
        # Convert to list and normalize
        embedding_list = embedding.tolist()
        
        return embedding_list


class EmbeddingHandler(BaseHTTPRequestHandler):
    """HTTP handler for embedding requests."""
    
    def __init__(self, *args, model=None, **kwargs):
        """Initialize with model."""
        self.model = model
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path != "/api/embeddings":
            self.send_error(404, "Not Found")
            return
        
        # Get content length
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_error(400, "Empty request")
            return
        
        # Read request body
        request_body = self.rfile.read(content_length).decode("utf-8")
        try:
            request_data = json.loads(request_body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        
        # Get prompt from request
        prompt = request_data.get("prompt")
        if not prompt:
            self.send_error(400, "Missing prompt")
            return
        
        # Generate embedding
        embedding = self.model.get_embedding(prompt)
        if embedding is None:
            self.send_error(500, "Failed to generate embedding")
            return
        
        # Prepare response
        response = {
            "embedding": embedding,
            "model": self.model.model_name
        }
        
        # Send response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))
    
    def log_message(self, format, *args):
        """Override to provide custom logging."""
        print(f"{self.client_address[0]} - {format % args}")


def run_server(model_name, port):
    """Run the embedding server."""
    # Load model
    model = EmbeddingModel(model_name)
    
    # Create handler with model
    def handler(*args, **kwargs):
        return EmbeddingHandler(*args, model=model, **kwargs)
    
    # Start server
    server = HTTPServer(("0.0.0.0", port), handler)
    print(f"Starting server on port {port}")
    print(f"API endpoint: http://localhost:{port}/api/embeddings")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run embedding server")
    parser.add_argument(
        "--model", 
        default="all-MiniLM-L6-v2",
        help="Model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to run server on (default: 8000)"
    )
    
    args = parser.parse_args()
    run_server(args.model, args.port)

