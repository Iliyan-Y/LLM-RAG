# syntax=docker/dockerfile:1.7
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# System deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    unzip \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install python deps
RUN pip install --upgrade pip \
  && pip install -r /app/requirements.txt

# Copy project files
COPY util/ /app/util/
COPY index_pdfs.py /app/index_pdfs.py
COPY query_rag.py /app/query_rag.py
COPY ws_server.py /app/ws_server.py
COPY vectorstore.zip /app/vectorstore.zip

# Provide default envs (can be overridden at runtime)
ENV VECTORSTORE_DIR=/app/vectorstore \
    SEMANTIC_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
    LLM_PROVIDER=ollama \
    OLLAMA_MODEL=gemma3:latest \
    TOTAL_CHUNK_CONTEXT=6 \
    LOGGING_ENABLED=false

# Unzip vectorstore if present
RUN mkdir -p ${VECTORSTORE_DIR} \
  && if [ -f /app/vectorstore.zip ]; then unzip -o /app/vectorstore.zip -d ${VECTORSTORE_DIR}; fi

# Pre-download embeddings to bake into image cache
# This prevents first-request downloads in production
RUN python -c "from sentence_transformers import SentenceTransformer; import os; m=os.getenv('SEMANTIC_MODEL_NAME','sentence-transformers/all-MiniLM-L6-v2'); SentenceTransformer(m)"

# Expose WS/HTTP port
EXPOSE 8000

# Default command starts FastAPI WebSocket server
CMD ["uvicorn", "ws_server:app", "--host", "0.0.0.0", "--port", "8000"]