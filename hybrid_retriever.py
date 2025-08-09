from typing import List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.chat_models import ChatOllama
from pydantic import Field, PrivateAttr


class HybridRetriever(BaseRetriever):
    # ✅ Declare these so Pydantic allows them
    db: object = Field(...)
    embeddings: object = Field(...)
    k: int = Field(default=6)
    rewrite_query: bool = Field(default=True)
    llm_model: object = Field(default=None)
    logging: bool = Field(default=False)

    # Private attributes for internal use (not part of Pydantic schema)
    _rewrite_prompt: PromptTemplate = PrivateAttr()
    _rewrite_chain: LLMChain = PrivateAttr()
    _hyde_prompt: PromptTemplate = PrivateAttr()
    _hyde_chain: LLMChain = PrivateAttr()
    _multiquery_prompt: PromptTemplate = PrivateAttr()
    _multiquery_chain: LLMChain = PrivateAttr()

    def __init__(self, db, embeddings, ollama_model="llama2", rewrite_query=True, k=6, **kwargs):
        super().__init__(
            db=db,
            embeddings=embeddings,
            k=k,
            rewrite_query=rewrite_query,
            llm_model=ChatOllama(model=ollama_model),
            logging=False,
            **kwargs
        )

        self._rewrite_prompt = PromptTemplate(
            input_variables=["question"],
            template="Rewrite this into a clear, standalone query for searching a knowledge base:\n\n{question}"
        )
        self._rewrite_chain = LLMChain(llm=self.llm_model, prompt=self._rewrite_prompt)

        self._hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are an expert researcher. Write a hypothetical, detailed answer "
                "to the question below, even if you are not sure of the real answer. "
                "Focus on facts, terminology, and relevant concepts.\n\nQuestion: {question}"
            )
        )
        self._hyde_chain = LLMChain(llm=self.llm_model, prompt=self._hyde_prompt)

        self._multiquery_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "Generate 3 rephrasings of the following search query. "
                "Include synonyms, alternate phrasings, and related terms.\n\nQuestion: {question}"
            )
        )
        self._multiquery_chain = LLMChain(llm=self.llm_model, prompt=self._multiquery_prompt)

    def _rewrite(self, query: str) -> str:
        if not self.rewrite_query:
            return query
        output = self._rewrite_chain.invoke(query)
        return output["text"].strip()

    def _hyde(self, query: str) -> str:
        output = self._hyde_chain.invoke(query)
        return output["text"].strip()

    def _multiquery(self, query: str) -> List[str]:
        output = self._multiquery_chain.invoke(query)
        variations_text = output["text"]
        variations = [v.strip("-• ").strip() for v in variations_text.split("\n") if v.strip()]
        return variations

    def _get_relevant_documents(self, query: str) -> List[Document]:
        if (self.rewrite_query):
            rewritten = self._rewrite(query)
            hyde_text = self._hyde(rewritten)
            variations = self._multiquery(hyde_text)
        else:
            variations = [query]

        if (self.logging):
            print(f"-  Rewritten query: {rewritten}")
            print(f"- Found {len(variations)} variations for hyde query: {hyde_text}")

        seen = set()
        all_docs = []
        score_threshold = 0.8  # adjust as needed
        
        for v in variations:
            results = self.db.similarity_search_with_score(v, k=self.k)  # returns (doc, score)
            for doc, score in results:
                if score >= score_threshold and doc.page_content not in seen:
                    all_docs.append(doc)
                    seen.add(doc.page_content)
        print(f"Found {len(all_docs)}")
        return all_docs
