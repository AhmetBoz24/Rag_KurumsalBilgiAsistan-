"""PDF metni icin temizleme yardimcilari.

PyMuPDF cikti tipik kirlilikleri temizler:
- Sayfa numaralari (1/18, 2/18 gibi)
- Satir sonunda kelime kirilmasi (intibak-\nlari -> intibaklari)
- Cok satirli paragraflarda anlamsiz satir sonlari (kelime + \n + kelime -> bosluk)
- Cift bosluklar, tab, BOM
"""
from __future__ import annotations
import re
import unicodedata


_PAGE_NUM_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.MULTILINE)
_TRAILING_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
_HYPHEN_BREAK_RE = re.compile(r"(\w+)-\n(\w+)")
_MID_PARAGRAPH_NL_RE = re.compile(r"(?<=[a-zçğıöşü,])\n(?=[a-zçğıöşü(])")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def clean_page_text(text: str) -> str:
    """Tek bir sayfa metnini temizler."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("﻿", "")

    text = _PAGE_NUM_RE.sub("", text)
    text = _TRAILING_PAGE_NUM_RE.sub("", text)

    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _MID_PARAGRAPH_NL_RE.sub(" ", text)

    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)

    return text.strip()


_MADDE_RE = re.compile(r"MADDE\s+(\d+)", re.IGNORECASE)
_BOLUM_RE = re.compile(r"^([A-ZÇĞİÖŞÜ]+)\s+BÖLÜM", re.MULTILINE)


def extract_madde(text: str) -> str | None:
    """Verilen metin parcasinda gecen ILK 'MADDE X'i dondurur."""
    m = _MADDE_RE.search(text)
    return f"MADDE {m.group(1)}" if m else None


def extract_bolum(text: str) -> str | None:
    """Metindeki son gorulen 'X BOLUM' basligini dondurur (chunk hangi bolume aitse)."""
    matches = _BOLUM_RE.findall(text)
    return f"{matches[-1]} BOLUM" if matches else None
