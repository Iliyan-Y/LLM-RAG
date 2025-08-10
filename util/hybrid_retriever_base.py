from typing import List, Optional, Tuple, Dict, Set
import re
from datetime import datetime
from collections import defaultdict

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.chat_models import ChatOllama
from pydantic import Field, PrivateAttr

class QueryProps:
  def __init__(self,  rewrite_query: bool, allow_multi_query: bool, allow_hyde: bool):
    self.rewrite_query = rewrite_query #Allow llm to rewrite the user prop for better semantic search
    self.allow_multi_query = allow_multi_query # Generate multiple variations of the query
    self.allow_hyde = allow_hyde # Generates hypothetical document based on the user's query

defaultQueryProps = QueryProps(rewrite_query=False, allow_multi_query=False, allow_hyde=False)

class HybridRetrieverBase(BaseRetriever):
    # Declare these so Pydantic allows them
    db: object = Field(...)
    embeddings: object = Field(...)
    k: int = Field(default=6)
    query_props: object = Field(default=defaultQueryProps)
    llm_model: object = Field(default=None)
    logging: bool = Field(default=False)

    # Private attributes for internal use (not part of Pydantic schema)
    _rewrite_prompt: PromptTemplate = PrivateAttr()
    _rewrite_chain: LLMChain = PrivateAttr()
    _hyde_prompt: PromptTemplate = PrivateAttr()
    _hyde_chain: LLMChain = PrivateAttr()
    _multiquery_prompt: PromptTemplate = PrivateAttr()
    _multiquery_chain: LLMChain = PrivateAttr()

    def __init__(self, db, embeddings, ollama_model="llama2", query_props= defaultQueryProps, k=6, logging=False, **kwargs):
        super().__init__(
            db=db,
            embeddings=embeddings,
            k=k,  # chunks to retrieve per query
            query_props=query_props,
            llm_model=ChatOllama(model=ollama_model),
            logging=logging,
            **kwargs
        )

        self._rewrite_prompt = PromptTemplate(
            input_variables=["question"],
            template="Convert the following prompt into a concise, self-contained query optimized for semantic search in a FAISS knowledge base. Keep all essential meaning, remove irrelevant context.: \n\n{question}"
        )
        self._rewrite_chain = LLMChain(llm=self.llm_model, prompt=self._rewrite_prompt)

        self._hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are a financial filings expert specializing in U.S. public company disclosures. "
                "Generate a hypothetical but detailed answer to the question below as if it were extracted from 10-Q or 10-K filings of NYSE-listed companies. "
                "Use realistic financial terminology, section headers, and language commonly found in SEC filings, including: "
                "Management's Discussion and Analysis (MD&A) "
                "Risk Factors "
                "Financial Statements and Notes "
                "Liquidity and Capital Resources "
                "Market Risk Disclosures "
                "Include relevant KPIs (EPS, net income, revenue, cash flow, total assets, liabilities, debt-to-equity ratio), dates, fiscal periods, industry terms, and compliance language. "
                "The goal is to create a richly detailed passage containing as many potentially relevant concepts and terms as possible to maximize matching with real filings in the knowledge base. "
                "Question: {question}"
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

    # --------- Helpers for time/type intent parsing and scoring ---------
    def _parse_temporal_intent(self, q: str) -> Dict:
        text = q.lower()
        # Flags
        prefer_latest = any(x in text for x in ["latest", "most recent", "newest", "current", "recent"])
        prefer_oldest = any(x in text for x in ["oldest", "earliest", "historical", "as of inception"])
        # Explicit year(s)
        years = [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
        target_year: Optional[int] = years[-1] if years else None
        # Quarter patterns: Q1/Q2/Q3/Q4 or "first/second/third/fourth quarter"
        qnum = None
        m = re.search(r"\bq([1-4])\b", text)
        if m:
            qnum = int(m.group(1))
        else:
            mapping = {"first": 1, "second": 2, "third": 3, "fourth": 4}
            m2 = re.search(r"\b(first|second|third|fourth)\s+quarter\b", text)
            if m2:
                qnum = mapping[m2.group(1)]
        # Date range like 2021-2023
        dr = None
        m3 = re.search(r"\b(20\d{2})\s*[-–]\s*(20\d{2})\b", text)
        if m3:
            a, b = int(m3.group(1)), int(m3.group(2))
            dr = (min(a, b), max(a, b))
        return {
            "prefer_latest": prefer_latest and not prefer_oldest,
            "prefer_oldest": prefer_oldest and not prefer_latest,
            "target_year": target_year,
            "target_quarter": qnum,
            "year_range": dr,
        }

    def _preferred_filing_type(self, q: str) -> Optional[str]:
        text = q.lower()
        if "10-k" in text or "annual report" in text or "annual" in text:
            return "10-K"
        if "10-q" in text or "quarterly" in text:
            return "10-Q"
        return None

    def _to_date(self, s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None

    def _time_score(self, doc: Document, intent: Dict, pool_dates: List[datetime]) -> float:
        # Use period_end_date primarily, fallback to filing_date
        md = doc.metadata or {}
        d = self._to_date(md.get("period_end_date")) or self._to_date(md.get("filing_date"))
        if not d:
            return 0.0

        # Direct matches by year/quarter
        ty = intent.get("target_year")
        tq = intent.get("target_quarter")
        yr = d.year
        score = 0.0

        if ty:
            if yr == ty:
                score += 0.8
            elif abs(yr - ty) == 1:
                score += 0.4
        if tq:
            # Approx quarter from month
            qm = (d.month - 1) // 3 + 1
            if qm == tq:
                score += 0.3

        yr_range = intent.get("year_range")
        if yr_range:
            a, b = yr_range
            if a <= yr <= b:
                score += 0.6

        # Recency/antiquity preference
        if pool_dates:
            mn = min(pool_dates)
            mx = max(pool_dates)
            if mx > mn:
                recency = (d - mn).total_seconds() / max(1.0, (mx - mn).total_seconds())
            else:
                recency = 0.5
        else:
            recency = 0.5

        if intent.get("prefer_latest"):
            score += recency * 0.7
        if intent.get("prefer_oldest"):
            score += (1.0 - recency) * 0.7

        return min(score, 1.5)

    def _type_score(self, doc: Document, pref: Optional[str]) -> float:
        if not pref:
            return 0.0
        ft = (doc.metadata or {}).get("filing_type")
        return 0.3 if ft and ft.upper() == pref.upper() else 0.0

    def _rank_score(self, rank_idx: int, max_k: int) -> float:
        # Convert rank position into a score in [0,1]
        return 1.0 - (rank_idx / max(1, max_k - 1))

    def _diversify(self, docs_scored: List[Tuple[Document, float]], k: int, per_source_cap: int = 2) -> List[Document]:
        out: List[Document] = []
        per_source = defaultdict(int)
        for doc, _ in docs_scored:
            src = (doc.metadata or {}).get("source") or (doc.metadata or {}).get("source_file") or "unknown"
            if per_source[src] >= per_source_cap and len(out) < k:
                continue
            out.append(doc)
            per_source[src] += 1
            if len(out) >= k:
                break
        # If we couldn't fill k due to caps, relax caps and fill remaining
        if len(out) < k:
            for doc, _ in docs_scored:
                if doc in out:
                    continue
                out.append(doc)
                if len(out) >= k:
                    break
        return out[:k]

    def _extract_company_hints(self, q: str) -> Dict[str, Set[str]]:
        """
        Extract potential company identifiers (tickers and names) from the query text.
        - Tickers: 1-6 uppercase letters, excluding common short words.
        - Names: Title-cased tokens (e.g., 'Meta', 'Tesla') and common two-token company names if present.
        Returns lowercase hints for matching.
        """
        tickers: Set[str] = set()
        names: Set[str] = set()

        text = q.strip()
        # Extract potential tickers (upper-case tokens)
        ticker_candidates = re.findall(r"\b[A-Z]{1,6}\b", text)
        stop = {
            "AND", "OR", "THE", "FOR", "WITH", "WHAT", "HOW", "ARE", "IS", "TO", "IN", "ON",
            "Q", "Q1", "Q2", "Q3", "Q4", "FY", "SEC", "USD", "EPS", "GAAP", "NONGAAP",
            "AS", "OF", "VS", "AN", "A", "BY", "FROM", "AT"
        }
        for t in ticker_candidates:
            if t not in stop and not re.fullmatch(r"20\d{2}", t):
                tickers.add(t.lower())

        # Extract simple TitleCase single-token names (e.g., Meta, Tesla)
        name_candidates = re.findall(r"\b([A-Z][a-z]+)\b", text)
        common_noise = {"What", "How", "And", "Or", "For", "With", "The", "Vs", "As", "Of", "In", "On", "Between", "Compared"}
        for n in name_candidates:
            if n not in common_noise and len(n) > 2:
                names.add(n.lower())

        # Extract two-token company names (e.g., 'Meta Platforms', 'Morgan Stanley')
        two_token_candidates = re.findall(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", text)
        for a, b in two_token_candidates:
            pair = f"{a} {b}".lower()
            if a not in common_noise and b not in common_noise:
                names.add(pair)

        return {"tickers": tickers, "names": names}

    def company_match(self, md: Dict, hints: Dict[str, Set[str]]) -> bool:
        """
        Return True if the document's metadata (or source filename) indicates it belongs
        to the target company derived from the query.
        Checks keys: company, category, ticker, cik, source, source_file, source_document, file_name.
        Performs case-insensitive substring checks against extracted tickers/names.
        """
        if not hints or (not hints.get("tickers") and not hints.get("names")):
            return False

        tickers = hints.get("tickers", set())
        names = hints.get("names", set())

        md = md or {}
        keys = ("company", "category", "ticker", "cik", "source", "source_file", "source_document", "file_name")
        haystacks: List[str] = []
        for k in keys:
            v = md.get(k)
            if isinstance(v, (str,)):
                haystacks.append(v.lower())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, str):
                        haystacks.append(item.lower())

        for h in tickers.union(names):
            for hay in haystacks:
                if h in hay:
                    return True
        return False