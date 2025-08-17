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
from langchain.memory import ConversationBufferWindowMemory
from contextlib import asynccontextmanager

load_dotenv()

# Environment configuration
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "/app/vectorstore")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")
SEMANTIC_MODEL_NAME = os.getenv("SEMANTIC_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # 'ollama' or 'openai'
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOTAL_CHUNK_CONTEXT = int(os.getenv("TOTAL_CHUNK_CONTEXT", "6"))
LOGGING_ENABLED = os.getenv("LOGGING_ENABLED", "false").lower() == "true"
MEMORY_BUF_SIZE = int(os.getenv("MEMORY_BUF_SIZE", 4)) # Number of previous messages 0 = off
 
# Concurrency control for LLM/RAG executions (tune via env MAX_CONCURRENT_LLM)
MAX_CONCURRENT_LLM = int(os.getenv("MAX_CONCURRENT_LLM", "4"))
 
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

# Global references for building per-session chains (to avoid shared memory)
retriever_global: Optional[HybridRetriever] = None
llm_global: Optional[Any] = None

# Semaphore to limit concurrent LLM/RAG executions
llm_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)


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
 
    # Save components globally so we can create per-session chains with isolated memory
    global retriever_global, llm_global
    retriever_global = retriever
    llm_global = llm
 
    # RetrievalQA Chain without shared memory (per-session memory will be created per connection)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain
    qa_chain =  init_rag_chain()
    print(f"{bcolors.OKGREEN}RAG WS server initialized. Provider={LLM_PROVIDER}{bcolors.ENDC}")
    yield
    # Clean up resources here if needed

app = FastAPI(lifespan=lifespan)


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
        # Offload potentially-blocking invoke() to a thread and respect concurrency limits
        async with llm_semaphore:
            response = await asyncio.to_thread(qa_chain.invoke, query)
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
    # Create per-connection memory and session-specific chain so conversations are isolated
    global retriever_global, llm_global, qa_chain
    if not qa_chain or not retriever_global or not llm_global:
        await websocket.send_text(json.dumps({"type":"error","error":"RAG not initialized"}))
        await websocket.close()
        return
    # Effect: Each user gets private conversation memory (k=4). For 100 concurrent users, memory usage increases (roughly proportional to active sessions × messages stored); consider an external memory store or session eviction for scale.
    if (MEMORY_BUF_SIZE > 0):
     memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="query", output_key="result", k=MEMORY_BUF_SIZE, return_messages=True)
     session_chain = RetrievalQA.from_chain_type(llm=llm_global, retriever=retriever_global, return_source_documents=True, memory=memory)
    else:
     session_chain = RetrievalQA.from_chain_type(llm=llm_global, retriever=retriever_global, return_source_documents=True)
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
                # Execute RAG (offload to threadpool and respect concurrency limit) using session-specific chain
                async with llm_semaphore:
                    response = await asyncio.to_thread(session_chain.invoke, query)
                if LOGGING_ENABLED:
                    print(f"{bcolors.OKBLUE}Answer: {response['result']} {bcolors.ENDC}")
                    for doc in response["source_documents"]:
                        print(f"Source: {doc.metadata['source']}, filing_type: {doc.metadata['filing_type']}, period: {doc.metadata['period_end_date']}, page: {doc.metadata['page_label']}")

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