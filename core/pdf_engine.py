"""PDF okuma motoru.

PyMuPDF (fitz) ile her PDF'i sayfa bazinda okur, temizler ve
metadata tasiyan yapilara donusturur.

Kullanim:
    from core.pdf_engine import extract_pages, scan_pdf_dir

    pages = extract_pages("data/58336.pdf")
    # -> [{"source": "58336.pdf", "page": 1, "text": "..."}, ...]
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Iterable

import fitz

from core.text_cleaner import clean_page_text


def extract_pages(pdf_path: str | Path) -> List[Dict]:
    """Tek bir PDF'i sayfa bazinda okur ve temiz metin + metadata dondurur."""
    pdf_path = Path(pdf_path)
    pages: List[Dict] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[pdf_engine] PDF acilamadi {pdf_path.name}: {e}")
        return pages

    print(f"[pdf_engine] {pdf_path.name} -> {len(doc)} sayfa")

    for i, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        cleaned = clean_page_text(raw)
        if len(cleaned) < 20:
            continue
        pages.append(
            {
                "source": pdf_path.name,
                "page": i,
                "text": cleaned,
            }
        )

    doc.close()
    return pages


def scan_pdf_dir(data_dir: str | Path) -> Iterable[Path]:
    """data/ klasorundeki tum PDF dosyalarini dondurur."""
    data_dir = Path(data_dir)
    return sorted(data_dir.glob("*.pdf"))


def main():
    """CLI: data/ klasorundeki tum PDF'leri okuyup ozet basar."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"

    pdfs = list(scan_pdf_dir(data_dir))
    if not pdfs:
        print(f"[pdf_engine] {data_dir} altinda PDF yok.")
        return

    print(f"[pdf_engine] {len(pdfs)} PDF bulundu.")
    total_pages = 0
    total_chars = 0
    for p in pdfs:
        pages = extract_pages(p)
        total_pages += len(pages)
        total_chars += sum(len(pg["text"]) for pg in pages)
        print(f"  - {p.name}: {len(pages)} dolu sayfa")

    print(
        f"\n[pdf_engine] Toplam: {total_pages} sayfa, "
        f"{total_chars:,} karakter."
    )


if __name__ == "__main__":
    main()
