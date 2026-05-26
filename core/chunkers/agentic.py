"""Agentic Chunker: basliklar + MADDE'lere duyarli, metadata uretir.

Girdi: sayfa metni
Cikti: List[Dict] -> [{"text": "...", "metadata": {...}}, ...]
"""
from __future__ import annotations
from typing import List, Dict
import re

from core.chunkers.sentence_based import run as sentence_chunk
from core.text_cleaner import extract_madde, extract_bolum


_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\s+[A-ZÇĞİÖŞÜ]")
_MADDE_HEADING_RE = re.compile(r"^MADDE\s+\d+", re.IGNORECASE)
_BOLUM_HEADING_RE = re.compile(r"^[A-ZÇĞİÖŞÜ]+\s+BÖLÜM", re.IGNORECASE)


def _is_heading(s: str) -> bool:
    s = s.strip()
    if _NUMBERED_HEADING_RE.match(s):
        return True
    if _MADDE_HEADING_RE.match(s):
        return True
    if _BOLUM_HEADING_RE.match(s):
        return True
    if s.isupper() and 2 <= len(s.split()) <= 10 and not s.endswith((".", ":", ";", ",", "!")):
        return True
    return False


def chunk_page(
    page_text: str,
    base_metadata: Dict,
    target_chars: int = 1000,
    overlap_sent: int = 1,
    current_bolum: str | None = None,
) -> List[Dict]:
    """Bir sayfanin metnini agentic olarak chunk'lar, metadata atar.

    base_metadata: {"source": "...", "page": N}
    Donen her chunk metadata'sina madde ve bolum bilgisi eklenir.
    """
    lines = [ln for ln in page_text.split("\n") if ln.strip()]

    sections: List[List[str]] = []
    cur: List[str] = []
    for ln in lines:
        if _is_heading(ln):
            if cur:
                sections.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        sections.append(cur)

    if not sections:
        sections = [lines]

    out: List[Dict] = []
    bolum = current_bolum

    for sec in sections:
        sec_text = "\n".join(sec).strip()
        if not sec_text:
            continue

        new_bolum = extract_bolum(sec_text)
        if new_bolum:
            bolum = new_bolum

        sec_chunks = sentence_chunk(sec_text, target_chars=target_chars, overlap_sent=overlap_sent)
        for ch in sec_chunks:
            md = dict(base_metadata)
            madde = extract_madde(ch)
            if madde:
                md["madde"] = madde
            if bolum:
                md["bolum"] = bolum
            out.append({"text": ch, "metadata": md})

    return out


def run(
    pages: List[Dict],
    target_chars: int = 1000,
    overlap_sent: int = 1,
) -> List[Dict]:
    """Sayfa listesini al, hepsini chunk'la, metadata ile dondur.

    pages: [{"source": "...", "page": N, "text": "..."}]
    return: [{"text": "...", "metadata": {source, page, madde?, bolum?}}]
    """
    all_chunks: List[Dict] = []
    bolum_state: str | None = None

    for pg in pages:
        base_md = {"source": pg["source"], "page": pg["page"]}
        chunks = chunk_page(
            pg["text"],
            base_md,
            target_chars=target_chars,
            overlap_sent=overlap_sent,
            current_bolum=bolum_state,
        )
        for c in chunks:
            if "bolum" in c["metadata"]:
                bolum_state = c["metadata"]["bolum"]
        all_chunks.extend(chunks)

    return all_chunks
