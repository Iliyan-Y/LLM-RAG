# syntax=docker/dockerfile:1.7
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1

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
  && pip install -r /app/requirements.txt \
  && pip install "huggingface_hub[cli]>=0.23"

# Copy project files
COPY util/ /app/util/
COPY index_pdfs.py /app/index_pdfs.py
COPY query_rag.py /app/query_rag.py
COPY ws_server.py /app/ws_server.py
COPY vectorstore/ /app/vectorstore/
COPY .env /app/.env

RUN huggingface-cli download "intfloat/e5-base-v2"

# Expose WS/HTTP port
EXPOSE 8000

# Default command starts FastAPI WebSocket server
CMD ["uvicorn", "ws_server:app", "--host", "0.0.0.0", "--port", "8000"]