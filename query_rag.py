import os
import sys
from util.hybrid_retriever import HybridRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from util.colors import bcolors
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

from util.hybrid_retriever_base import QueryProps
load_dotenv()

VECTORSTORE_DIR = os.getenv('VECTORSTORE_DIR')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')
SEMANTIC_MODEL_NAME = os.getenv('SEMANTIC_MODEL_NAME')
TOTAL_CHUNK_CONTEXT = 6

# Options for narrowing the sematic search scope
qp = QueryProps(rewrite_query=True, allow_multi_query=False, allow_hyde=False)

if not os.path.exists(VECTORSTORE_DIR):
    print(f"{bcolors.FAIL}Vectorstore directory '{VECTORSTORE_DIR}' not found. Run index_pdfs.py first.{bcolors.ENDC}")
    sys.exit(1)

embeddings = HuggingFaceEmbeddings(model_name=SEMANTIC_MODEL_NAME)#all-mpnet-base-v2 all-MiniLM-L6-v2
db = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

 
retriever = HybridRetriever(db=db, embeddings=embeddings, ollama_model=OLLAMA_MODEL, query_props=qp, k=TOTAL_CHUNK_CONTEXT, logging=True) 

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
        print(f"{bcolors.OKBLUE}Answer: {response["result"]} {bcolors.ENDC}\n",)
        for doc in response["source_documents"]:
            print(f"Source: {doc.metadata['source']}, filing_type: {doc.metadata['filing_type']}, period: {doc.metadata['period_end_date']}, page: {doc.metadata['page_label']}")
        os.system('afplay /System/Library/Sounds/Hero.aiff')
    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except Exception as e:
        print(f"{bcolors.FAIL}Error during query: {e}{bcolors.ENDC}")