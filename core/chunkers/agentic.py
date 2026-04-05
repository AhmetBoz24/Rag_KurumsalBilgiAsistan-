from typing import List
import re
from core.chunkers.sentence_based import run as sentence_chunk

# Baslık sezgisi (section detection)
def _is_heading(s: str) -> bool:
    s_stripped = s.strip()

    # 1) Numaralı başlıklar: 1, 1.2, 3.4.5 vb. (En güvenilir başlık tipi)
    if re.match(r"^\d+(?:\.\d+)*\s+[A-ZÇĞİÖŞÜ]", s_stripped):
        return True

    # 2) Tümü büyük harf başlık (Ancak çok uzun değilse ve noktalama yoksa)
    if s_stripped.isupper() and 2 <= len(s_stripped.split()) <= 10:
        if not s_stripped.endswith(('.', ':', ';', ',', '!')):
            return True

    # 3) 'MADDE X' içeren satırlar
    if re.match(r"^MADDE\s+\d+", s_stripped, re.IGNORECASE):
        return True

    return False


def run(
    text: str,
    target_chars: int = 1200,
    overlap_sent: int = 0,
) -> List[str]:
    """
    Agentic Chunking:
      1) Başlıkları yakala ve metni bölümlere ayır.
      2) Her bölüm içinde cümle bazlı chunk'la (sentence_based).
      3) Başlık yoksa, komple metni cümle bazlı chunk'la (fallback).

    Parametreler:
      - target_chars: bölüm içi chunk hedef uzunluğu
      - overlap_sent: bölüm içindeki sentence-based overlap
    """
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in norm.split("\n") if ln.strip()]

    # Bölümleme: heading gördükçe yeni section başlat
    sections: List[List[str]] = []
    cur: List[str] = []
    for ln in lines:
        if _is_heading(ln):
            if cur:
                sections.append(cur)
                cur = []
        cur.append(ln)
    if cur:
        sections.append(cur)

    # Başlık hiç yakalanmadıysa fallback: tüm metni sentence-based yap
    if not sections:
        return sentence_chunk(text, target_chars=target_chars, overlap_sent=overlap_sent)

    # Her bölüm içinde sentence-based; başlığı öncelikli olarak dahil et
    chunks: List[str] = []
    for sec in sections:
        # Bölüm metni: (başlık + içerik) -> tek parça string
        # Başlığı içerikte ilk cümle olarak bırakmak, retrieval'da bağlamı güçlendirir
        sec_text = "\n".join(sec).strip()

        # Bölüm içi cümle bazlı chunk
        sec_chunks = sentence_chunk(sec_text, target_chars=target_chars, overlap_sent=overlap_sent)

        # (Opsiyonel) Çok kısa tekil başlık chunk'larını bir sonrakine birleştirmek istersek
        # burada ek bir kural koyabiliriz. Minimal versiyonda direkt ekliyoruz.
        chunks.extend(sec_chunks)

    return chunks