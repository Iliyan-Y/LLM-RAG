from typing import List, Tuple
from datetime import datetime
from langchain.schema import Document
from util.colors import bcolors
from util.hybrid_retriever_base import HybridRetrieverBase

class HybridRetriever(HybridRetrieverBase):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1) Query rewriting + HYDE + multiquery
        if self.rewrite_query:
            rewritten = self._rewrite(query)
            hyde_text = self._hyde(rewritten)
            variations = self._multiquery(hyde_text)
        else:
            rewritten = query
            variations = [query]

        if self.logging and self.rewrite_query:
            print(f"{bcolors.OKGREEN}Found {len(variations)} variations{bcolors.ENDC}")
            print(f"{bcolors.WARNING}- Rewritten: {rewritten}{bcolors.ENDC}")
            print(f"{bcolors.OKGREEN}- HYDE: {hyde_text}{bcolors.ENDC}")
            for v in variations:
                print(f"{bcolors.HEADER}- Variation: {v}{bcolors.ENDC}")

        # Derive company hints from the (rewritten) user query
        company_hints = self._extract_company_hints(rewritten)
        if self.logging and (company_hints["tickers"] or company_hints["names"]):
            print(f"{bcolors.OKGREEN}- Company hints detected: tickers={sorted(company_hints['tickers'])}, names={sorted(company_hints['names'])}{bcolors.ENDC}")

        # 2) Gather candidate pool (rank-based scoring to avoid distance sign issues)
        candidate_pool: List[Tuple[Document, int]] = []  # (doc, rank_index)
        seen_keys = set()
        fetch_k = max(self.k * 4, 12)

        for v in variations:
            results = self.db.similarity_search_with_score(v, k=fetch_k)
            for rank, (doc, _raw_score) in enumerate(results):
                key = ((doc.metadata or {}).get("source"), (doc.metadata or {}).get("chunk_index"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidate_pool.append((doc, rank))

        if self.logging:
            print(f"{bcolors.OKGREEN}- Candidate pool size (pre-company filter): {len(candidate_pool)}.{bcolors.ENDC}")

        if not candidate_pool:
            return []

        # 2b) Optional company filtering: keep only candidates that match company/ticker/category hints
        if company_hints["tickers"] or company_hints["names"]:
            filtered_pool = [(doc, r) for (doc, r) in candidate_pool if self.company_match(doc.metadata, company_hints)]
            if filtered_pool:
                candidate_pool = filtered_pool
                if self.logging:
                    print(f"{bcolors.OKGREEN}- Candidate pool size (post-company filter): {len(candidate_pool)}.{bcolors.ENDC}")
            elif self.logging:
                print(f"{bcolors.WARNING}- No company-matching candidates found; falling back to unfiltered pool.{bcolors.ENDC}")

        # 3) Temporal and filing-type intent
        intent = self._parse_temporal_intent(rewritten)
        type_pref = self._preferred_filing_type(rewritten)

        # Collect dates for normalization
        pool_dates: List[datetime] = []
        for doc, _ in candidate_pool:
            d = self._to_date((doc.metadata or {}).get("period_end_date")) or self._to_date((doc.metadata or {}).get("filing_date"))
            if d:
                pool_dates.append(d)

        # 4) Score each candidate: final = w_sem*rank + w_time*time + w_type*type
        w_sem, w_time, w_type = 0.55, 0.35, 0.10
        max_rank_k = max(1, min(fetch_k, 50))

        scored: List[Tuple[Document, float]] = []
        for doc, rank_idx in candidate_pool:
            rscore = self._rank_score(rank_idx, max_rank_k)
            tscore = self._time_score(doc, intent, pool_dates)
            tyscore = self._type_score(doc, type_pref)
            final = w_sem * rscore + w_time * tscore + w_type * tyscore
            scored.append((doc, final))

        # 5) Sort and diversify across sources
        scored.sort(key=lambda x: x[1], reverse=True)
        diversified = self._diversify(scored, self.k, per_source_cap=2)
        if (self.logging):
            print(f"{bcolors.OKGREEN}- total scored: {len(scored)}, Selected  after diversification: {len(diversified)} docs{bcolors.ENDC}")

        if self.logging:
            sources = [(d.metadata or {}).get("source") for d in diversified]
            print(f"{bcolors.OKGREEN}- Selected {len(diversified)} docs from sources: {sources}{bcolors.ENDC}")

        return diversified
