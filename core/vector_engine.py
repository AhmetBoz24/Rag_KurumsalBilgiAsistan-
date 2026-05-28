"""Vektor veritabani insa motoru.

Iki kosumu vardir:
1) build_or_update(): hash bazli incremental — sadece yeni/degisen PDF'leri reindex eder
2) rebuild_all(): tam sifirdan

Her iki durumda da hem Chroma (dense) hem BM25 indeksini gunceller.
"""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set

from langchain_chroma import Chroma

from core.config import (
    CHROMA_DIR,
    DATA_DIR,
    HASH_PATH,
    CHUNK_TARGET_CHARS,
    CHUNK_OVERLAP_SENT,
)
from core.embeddings import get_embeddings
from core.pdf_engine import extract_pages, scan_pdf_dir
from core.chunkers.agentic import run as agentic_chunk
from core.retriever import save_bm25, tokenize_tr  # noqa: F401


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_hashes() -> Dict[str, str]:
    if HASH_PATH.exists():
        with open(HASH_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_hashes(hashes: Dict[str, str]) -> None:
    HASH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HASH_PATH, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)


def _diff(current: Dict[str, str], previous: Dict[str, str]):
    """(yeni_veya_degisen, silinen) PDF isim setlerini dondurur."""
    new_or_changed: Set[str] = set()
    for name, h in current.items():
        if previous.get(name) != h:
            new_or_changed.add(name)
    removed: Set[str] = set(previous.keys()) - set(current.keys())
    return new_or_changed, removed


def _chunk_pdfs(pdf_paths: List[Path]) -> List[Dict]:
    """Verilen PDF'leri okuyup chunk'lar (metadata ile)."""
    all_chunks: List[Dict] = []
    for p in pdf_paths:
        pages = extract_pages(p)
        chunks = agentic_chunk(
            pages,
            target_chars=CHUNK_TARGET_CHARS,
            overlap_sent=CHUNK_OVERLAP_SENT,
        )
        print(f"  -> {p.name}: {len(chunks)} chunk")
        all_chunks.extend(chunks)
    return all_chunks


def _persist_chroma_full(chunks: List[Dict]) -> None:
    """Chroma'yi sifirla ve yeniden doldur.

    NOT: filesystem'i silmek yerine collection'i bosaltiyoruz, cunku
    API canli oldugunda SQLite dosyasi locked olabilir (Windows).
    """
    print(f"[vector_engine] Embedding modeli yukleniyor...")
    embeddings = get_embeddings()

    vs = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    # Mevcut tum chunk'lari sil
    try:
        existing = vs.get()
        ids = existing.get("ids") if existing else None
        if ids:
            print(f"[vector_engine] Eski indeks temizleniyor ({len(ids)} chunk)...")
            vs.delete(ids=ids)
    except Exception as e:
        print(f"[vector_engine] Soft delete uyarisi: {e}")

    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    if not texts:
        print("[vector_engine] Eklenecek chunk yok.")
        return

    print(f"[vector_engine] {len(texts)} chunk Chroma'ya yaziliyor...")
    vs.add_texts(texts=texts, metadatas=metas)


def _persist_chroma_incremental(
    add_chunks: List[Dict], remove_sources: Set[str]
) -> None:
    """Chroma'da silinen kaynaklari temizle, yenileri ekle."""
    embeddings = get_embeddings()
    vs = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    if remove_sources:
        for src in remove_sources:
            vs.delete(where={"source": src})
            print(f"  -> {src} Chroma'dan silindi")

    if add_chunks:
        texts = [c["text"] for c in add_chunks]
        metas = [c["metadata"] for c in add_chunks]
        vs.add_texts(texts=texts, metadatas=metas)
        print(f"  -> {len(texts)} yeni chunk Chroma'ya eklendi")


def _rebuild_bm25_from_chroma() -> None:
    """Tum Chroma icerigini cekip BM25'i bastan kurar."""
    embeddings = get_embeddings()
    vs = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    data = vs.get()
    texts = data.get("documents", [])
    metas = data.get("metadatas", [])
    if not texts:
        print("[vector_engine] BM25 icin chunk yok.")
        return
    save_bm25(texts, metas)
    print(f"[vector_engine] BM25 indeksi: {len(texts)} chunk")


def build_or_update(force: bool = False) -> Dict:
    """Hash'leri karsilastir, sadece degisenleri reindex et.

    force=True ise tam rebuild.
    Donen rapor: {added, changed, removed, total}
    """
    pdfs = list(scan_pdf_dir(DATA_DIR))
    if not pdfs:
        print(f"[vector_engine] {DATA_DIR} altinda PDF yok.")
        return {"added": [], "changed": [], "removed": [], "total": 0}

    current_hashes = {p.name: _sha256_file(p) for p in pdfs}
    previous_hashes = {} if force else _load_hashes()

    new_or_changed, removed = _diff(current_hashes, previous_hashes)
    added = [n for n in new_or_changed if n not in previous_hashes]
    changed = [n for n in new_or_changed if n in previous_hashes]

    print(f"[vector_engine] Yeni: {len(added)}, Degisen: {len(changed)}, Silinen: {len(removed)}")

    chroma_exists = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())

    if force or not chroma_exists or not previous_hashes:
        print("[vector_engine] Tam rebuild...")
        chunks = _chunk_pdfs(pdfs)
        _persist_chroma_full(chunks)
    else:
        if not new_or_changed and not removed:
            print("[vector_engine] Degisiklik yok, indeks guncel.")
            return {"added": [], "changed": [], "removed": [], "total": len(pdfs)}

        changed_paths = [p for p in pdfs if p.name in new_or_changed]
        # Degisen PDF'lerin eski chunk'larini Chroma'dan sil, yenilerini ekle
        sources_to_remove = set(changed) | removed
        new_chunks = _chunk_pdfs(changed_paths) if changed_paths else []
        _persist_chroma_incremental(new_chunks, sources_to_remove)

    _rebuild_bm25_from_chroma()
    _save_hashes(current_hashes)

    return {
        "added": added,
        "changed": changed,
        "removed": list(removed),
        "total": len(pdfs),
    }


def rebuild_all() -> Dict:
    return build_or_update(force=True)


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    report = build_or_update(force=force)
    print("\n[vector_engine] OZET:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
