"""
Query RAG pipeline: Retrieve from FAISS vector DB and ask LLM (Ollama).

Usage:
    python3 query_rag.py

Requirements:
    pip install langchain faiss-cpu sentence-transformers ollama

- Ensure Ollama is running locally with a supported model pulled (e.g., llama3, gemma, etc.)
- The vector DB must exist in ./vectorstore/ (run index_pdfs.py first).
"""

import os
import sys
from hybrid_retriever import HybridRetriever

VECTORSTORE_DIR = "./vectorstore"
OLLAMA_MODEL = "gemma3:latest"  # Change to your preferred model

# 1. Load vector DB
try:
    from langchain.vectorstores import FAISS
except ImportError:
    print("Missing faiss-cpu. Install with: pip install faiss-cpu")
    sys.exit(1)

try:
    from langchain.embeddings import HuggingFaceEmbeddings
except ImportError:
    print("Missing langchain or sentence-transformers. Install with: pip install langchain sentence-transformers")
    sys.exit(1)

if not os.path.exists(VECTORSTORE_DIR):
    print(f"Vectorstore directory '{VECTORSTORE_DIR}' not found. Run index_pdfs.py first.")
    sys.exit(1)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

# default retriever
# retriever = db.as_retriever()
retriever = HybridRetriever(db=db, embeddings=embeddings, ollama_model=OLLAMA_MODEL, rewrite_query=True, k=6)

# 2. LLM (Ollama)
try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    print("Missing langchain or ollama. Install with: pip install langchain ollama")
    sys.exit(1)

llm = ChatOllama(model=OLLAMA_MODEL)

# 3. RetrievalQA Chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# 4. Query loop
print("RAG Query Interface. Type your question and press Enter (Ctrl+C to exit).")
while True:
    try:
        query = input("\nYour question: ").strip()
        if not query:
            continue
        response = qa_chain.invoke(query)
        print("Answer:", response["result"])
        for doc in response["source_documents"]:
            print(f"Source: {doc.metadata['source']}, Chunk: {doc.metadata['chunk_index']}")
    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except Exception as e:
        print("Error during query:", e)