from typing import List
import re

def _sent_tokenize(text: str) -> List[str]:
    """Cümlelere böl (önce nltk, yoksa regex)."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        # basit regex fallback
        return re.split(r'(?<=[.!?])\s+(?=[A-ZİÖÜÇĞŞ])', text)

def run(text: str, target_chars: int = 900, overlap_sent: int = 1) -> List[str]:
    """Cümle bazlı chunking: hedef uzunluğa ulaşana kadar cümleleri birleştirir."""
    sents = [s.strip() for s in _sent_tokenize(text) if s.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for s in sents:
        if len(s) > target_chars:
            # çok uzun cümle → direkt parça parça kes
            i = 0
            while i < len(s):
                end = min(i + target_chars, len(s))
                part = s[i:end].strip()
                if part:
                    if cur:
                        chunks.append(" ".join(cur))
                        cur = []
                        cur_len = 0
                    chunks.append(part)
                i = end
            continue

        if cur_len + len(s) + 1 <= target_chars:
            cur.append(s)
            cur_len += len(s) + 1
        else:
            chunks.append(" ".join(cur))
            # overlap: son birkaç cümleyi taşı
            cur = cur[-overlap_sent:] if overlap_sent > 0 else []
            cur.append(s)
            cur_len = sum(len(x) + 1 for x in cur)

    if cur:
        chunks.append(" ".join(cur))

    return chunks