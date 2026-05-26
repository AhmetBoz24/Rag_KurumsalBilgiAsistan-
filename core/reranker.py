"""BGE Reranker sarmalayicisi.

FlagEmbedding kutuphanesi yuklu degilse veya yukleme hata verirse
fallback olarak no-op reranker doner (sirayi degistirmez).
"""
from __future__ import annotations
from typing import List, Tuple

from core.config import ENABLE_RERANKER, RERANKER_MODEL, FINAL_TOP_K


class _NoopReranker:
    def rerank(self, query: str, docs: List[Tuple[str, dict]], top_k: int = FINAL_TOP_K):
        return docs[:top_k]


class BGEReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(model_name, use_fp16=True)

    def rerank(
        self,
        query: str,
        docs: List[Tuple[str, dict]],
        top_k: int = FINAL_TOP_K,
    ) -> List[Tuple[str, dict]]:
        if not docs:
            return []
        pairs = [[query, d[0]] for d in docs]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_k]]


def get_reranker():
    if not ENABLE_RERANKER:
        print("[reranker] Devre disi (ENABLE_RERANKER=false), no-op kullaniliyor.")
        return _NoopReranker()
    try:
        return BGEReranker()
    except Exception as e:
        print(f"[reranker] BGE yuklenemedi, no-op kullaniliyor: {e}")
        return _NoopReranker()
