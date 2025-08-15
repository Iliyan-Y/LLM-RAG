import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# LangChain / RAG components
from util.hybrid_retriever import HybridRetriever
from util.hybrid_retriever_base import QueryProps
from util.colors import bcolors
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

load_dotenv()

# Environment configuration
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "/app/vectorstore")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")
SEMANTIC_MODEL_NAME = os.getenv("SEMANTIC_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # 'ollama' or 'openai'
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOTAL_CHUNK_CONTEXT = int(os.getenv("TOTAL_CHUNK_CONTEXT", "6"))
LOGGING_ENABLED = os.getenv("LOGGING_ENABLED", "false").lower() == "true"

# Optional: let ollama python client discover a non-local host (docker-compose service)
# If not set, defaults to localhost
OLLAMA_HOST = os.getenv("OLLAMA_HOST")  # e.g. http://ollama:11434

# Global app and RAG state
app = FastAPI(title="RAG WebSocket Server", version="1.0.0")

# CORS to allow browsers to connect over the internet (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_chain: Optional[RetrievalQA] = None


def init_rag_chain() -> RetrievalQA:
    if not os.path.exists(VECTORSTORE_DIR):
        print(f"{bcolors.FAIL}Vectorstore directory '{VECTORSTORE_DIR}' not found. Provide it or unzip vectorstore.zip into it.{bcolors.ENDC}")
        sys.exit(1)

    # Embeddings (will download model on first use; for docker we attempt pre-download during build)
    embeddings = HuggingFaceEmbeddings(model_name=SEMANTIC_MODEL_NAME)

    # Load FAISS index
    db = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

    # Select LLM provider (Ollama local by default, or OpenAI via API)
    if LLM_PROVIDER == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            print(f"{bcolors.FAIL}langchain_openai not installed. pip install langchain-openai. Error: {e}{bcolors.ENDC}")
            sys.exit(1)
        if not os.getenv("OPENAI_API_KEY"):
            print(f"{bcolors.FAIL}OPENAI_API_KEY not set in environment. Set it in your .env file.{bcolors.ENDC}")
            sys.exit(1)
        llm = ChatOpenAI(model=OPENAI_MODEL)
    else:
        # If OLLAMA_HOST is set, the ollama client lib will pick it up
        if OLLAMA_HOST:
            os.environ["OLLAMA_HOST"] = OLLAMA_HOST
        llm = ChatOllama(model=OLLAMA_MODEL)

    # Options for narrowing the semantic search scope
    qp = QueryProps(rewrite_query=True, allow_multi_query=False, allow_hyde=False)

    # Build retriever with the chosen LLM (used internally for rewrite/hyde/multiquery)
    retriever = HybridRetriever(
        db=db,
        embeddings=embeddings,
        ollama_model=OLLAMA_MODEL,
        llm_model=llm,
        query_props=qp,
        k=TOTAL_CHUNK_CONTEXT,
        logging=LOGGING_ENABLED,
    )

    # RetrievalQA Chain
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return chain


def format_sources(source_documents: List[Any]) -> List[Dict[str, Any]]:
    formatted = []
    for doc in source_documents or []:
        meta = getattr(doc, "metadata", {}) or {}
        formatted.append(
            {
                "source": meta.get("source"),
                "filing_type": meta.get("filing_type"),
                "period_end_date": meta.get("period_end_date"),
                "page_label": meta.get("page_label"),
            }
        )
    return formatted


@app.on_event("startup")
async def on_startup():
    global qa_chain
    qa_chain = init_rag_chain()
    print(f"{bcolors.OKGREEN}RAG WS server initialized. Provider={LLM_PROVIDER}{bcolors.ENDC}")


@app.get("/health")
async def health():
    return {"status": "ok", "provider": LLM_PROVIDER, "model": OLLAMA_MODEL if LLM_PROVIDER == "ollama" else OPENAI_MODEL}


@app.post("/query")
async def http_query(payload: Dict[str, Any]):
    global qa_chain
    if not qa_chain:
        return JSONResponse(status_code=503, content={"error": "RAG not initialized"})
    query = (payload or {}).get("query", "").strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "query is required"})
    try:
        response = qa_chain.invoke(query)
        return {
            "answer": response.get("result"),
            "sources": format_sources(response.get("source_documents", [])),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "welcome", "message": "Connected to RAG WebSocket"}))
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw) if raw and raw.strip().startswith("{") else {"query": raw}
            except json.JSONDecodeError:
                data = {"query": raw}

            query = (data.get("query") or "").strip()
            if not query:
                await websocket.send_text(json.dumps({"type": "error", "error": "query is required"}))
                continue

            # Busy notification (optional)
            await websocket.send_text(json.dumps({"type": "status", "message": "processing"}))

            try:
                # Execute RAG
                response = qa_chain.invoke(query)
                message = {
                    "type": "answer",
                    "answer": response.get("result"),
                    "sources": format_sources(response.get("source_documents", [])),
                }
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
    except WebSocketDisconnect:
        # Client disconnected
        return
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
        except Exception:
            pass


if __name__ == "__main__":
    # For local run without docker: python ws_server.py
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("ws_server:app", host="0.0.0.0", port=port, reload=False)