import os
import sys
from hybrid_retriever import HybridRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from colors import bcolors
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

VECTORSTORE_DIR = "./vectorstore"
OLLAMA_MODEL = "gemma3:latest"  # Change to your preferred model

if not os.path.exists(VECTORSTORE_DIR):
    print(f"{bcolors.FAIL}Vectorstore directory '{VECTORSTORE_DIR}' not found. Run index_pdfs.py first.{bcolors.ENDC}")
    sys.exit(1)

embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")#all-mpnet-base-v2 all-MiniLM-L6-v2
db = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

# default retriever
# retriever = db.as_retriever()
retriever = HybridRetriever(db=db, embeddings=embeddings, ollama_model=OLLAMA_MODEL, rewrite_query=True, k=6, logging=True) # k is the number of chunks to retrieve

# 2. LLM (Ollama)
llm = ChatOllama(model=OLLAMA_MODEL)

# 3. RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# 4. Query loop
print(f"{bcolors.HEADER}RAG Query Interface. Type your question and press Enter (Ctrl+C to exit).{bcolors.ENDC}")
while True:
    try:
        query = input("\nYour question: ").strip()
        if not query:
            continue
        response = qa_chain.invoke(query)
        print(f"{bcolors.OKBLUE}Answer: {bcolors.ENDC}", response["result"])
        for doc in response["source_documents"]:
            print(f"Source: {doc.metadata['source']}, Chunk: {doc.metadata['chunk_index']}")
    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except Exception as e:
        print(f"{bcolors.FAIL}Error during query: {e}{bcolors.ENDC}")