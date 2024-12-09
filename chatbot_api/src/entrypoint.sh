#!/bin/bash

VECTOR_STORE_DOCKER_PATH="/app/vector_store"

# Debugging output
echo "Checking for vector store at $VECTOR_STORE_DOCKER_PATH"

# Check if the vector store exists and is not empty
if [ -d "$VECTOR_STORE_DOCKER_PATH" ] && [ "$(ls -A $VECTOR_STORE_DOCKER_PATH)" ]; then
    echo "Vector store found at $VECTOR_STORE_DOCKER_PATH. Skipping generation."
else
    echo "Vector store not found or is empty. Generating new vector store..."
    python /app/generate_vector_store.py
    echo "Vector store created successfully."
fi

# Run any setup steps or pre-processing tasks here
echo "Starting AskKhayrul RAG FastAPI service..."

# Start the main application
uvicorn main:app --host 0.0.0.0 --port 8000
