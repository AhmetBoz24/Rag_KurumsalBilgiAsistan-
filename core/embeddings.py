"""Embedding modeli sarmalayicisi.

ytu-ce-cosmos/turkish-e5-large -> Turkce icin yuksek kalite.
E5 modelleri 'passage:' / 'query:' prefix bekler — biz buna uyacak sekilde
LangChain'in HuggingFaceEmbeddings wrapper'ini kullanip prefix'i kendimiz ekliyoruz.
"""
from __future__ import annotations
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from core.config import EMBEDDING_MODEL


class TurkishE5Embeddings:
    """E5 ailesinin 'query:' / 'passage:' prefix gereksinimini saglar."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._inner = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return self._inner.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(f"query: {text}")


def get_embeddings() -> TurkishE5Embeddings:
    return TurkishE5Embeddings()
