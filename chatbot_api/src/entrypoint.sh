#!/bin/bash

# Generate the vector store
echo "Generating vector store..."
python /app/generate_vector_store.py

# Run any setup steps or pre-processing tasks here
echo "Starting AskKhayrul RAG FastAPI service..."

# Start the main application
uvicorn main:app --host 0.0.0.0 --port 8000