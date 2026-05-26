"""Hybrid Retriever: Dense (Chroma) + BM25 + RRF + Reranker.

Akis:
  query
   -> [dense_top_k]  (Chroma + Turkish E5)
   -> [bm25_top_k]   (rank_bm25)
   -> RRF birlestir  (k = RRF_K)
   -> Reranker (BGE) -> FINAL_TOP_K
"""
from __future__ import annotations
import os
import pickle
import re
import unicodedata
from typing import List, Tuple, Dict

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from core.config import (
    CHROMA_DIR,
    BM25_PATH,
    DENSE_TOP_K,
    BM25_TOP_K,
    RRF_K,
    FINAL_TOP_K,
)
from core.embeddings import get_embeddings
from core.reranker import get_reranker


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize_tr(text: str) -> List[str]:
    """Cok basit Turkce tokenizer (BM25 icin yeterli)."""
    text = unicodedata.normalize("NFKC", text).lower()
    return _TOKEN_RE.findall(text)


def normalize_query(q: str) -> str:
    """Madde N varyasyonlarini standart formata getir."""
    q = q.strip()
    q = re.sub(r"\b(\d+)\s*\.\s*madde\b", r"madde \1", q, flags=re.IGNORECASE)
    q = re.sub(r"\bmadde\s*(\d+)\b", r"madde \1", q, flags=re.IGNORECASE)
    return q


def save_bm25(corpus: List[str], metadatas: List[Dict], path=BM25_PATH) -> None:
    """BM25 indeksini disk'e kaydet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tokenized = [tokenize_tr(c) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus, "metadatas": metadatas}, f)


def load_bm25(path=BM25_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


class HybridRetriever:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embeddings,
        )
        self.bm25_data = load_bm25()
        self.reranker = get_reranker()

    def _dense_search(self, query: str, k: int) -> List[Tuple[str, Dict, float]]:
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(d.page_content, d.metadata, float(score)) for d, score in results]

    def _bm25_search(self, query: str, k: int) -> List[Tuple[str, Dict, float]]:
        if not self.bm25_data:
            return []
        bm25 = self.bm25_data["bm25"]
        corpus = self.bm25_data["corpus"]
        metas = self.bm25_data["metadatas"]
        toks = tokenize_tr(query)
        if not toks:
            return []
        scores = bm25.get_scores(toks)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(corpus[i], metas[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    @staticmethod
    def _rrf(
        dense: List[Tuple[str, Dict, float]],
        bm25: List[Tuple[str, Dict, float]],
        k: int = RRF_K,
    ) -> List[Tuple[str, Dict]]:
        """Reciprocal Rank Fusion. Skor: sum(1 / (k + rank))."""
        scores: Dict[str, float] = {}
        meta_lookup: Dict[str, Dict] = {}

        for rank, (text, md, _) in enumerate(dense):
            scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank)
            meta_lookup[text] = md
        for rank, (text, md, _) in enumerate(bm25):
            scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank)
            meta_lookup.setdefault(text, md)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(t, meta_lookup[t]) for t, _ in ranked]

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> List[Tuple[str, Dict]]:
        q = normalize_query(query)
        dense = self._dense_search(q, DENSE_TOP_K)
        bm25 = self._bm25_search(q, BM25_TOP_K)
        fused = self._rrf(dense, bm25)
        reranked = self.reranker.rerank(q, fused[:max(DENSE_TOP_K, BM25_TOP_K)], top_k=top_k)
        return reranked
