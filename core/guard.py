"""Prompt injection + cevap sizinti koruyucusu.

INPUT GUARD: Kullanicidan gelen mesaji LLM'e gondermeden once kontrol et.
OUTPUT GUARD: LLM'in cevabini kullaniciya gondermeden once kontrol et.
"""
from __future__ import annotations
import re
from typing import Optional


REJECT_MESSAGE = (
    "Ben Istanbul Gedik Universitesi'nin kurumsal bilgi asistaniyim. "
    "Sadece universite yonetmelikleri, sinav kurallari ve idari prosedurler "
    "hakkinda sorulariniza yardimci olabilirim."
)


# Kullanici girdisinde tehlike sinyalleri
_BLOCKED_INPUT_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instruction|prompt|rule)",
    r"forget\s+(?:everything|all|your)\s+(?:above|previous|instruction|rule)",
    r"disregard\s+(?:previous|all)\s+(?:instruction|prompt)",
    r"reveal\s+(?:your|the)?\s*(?:system\s+)?prompt",
    r"show\s+me\s+(?:your|the)?\s*(?:system\s+)?(?:prompt|instruction|rule)",
    r"what\s+(?:is|are)\s+your\s+(?:system\s+)?(?:prompt|instruction|rule)",
    r"sistem\s*prompt(?:un|unu|ungu|ungun|unuzu)",
    r"talimatlar(?:in(?:i|izi)|in|iniz)\s*(?:goster|paylas|soyle|yaz)",
    r"yonergeni\s*(?:goster|paylas|soyle|ac|yaz)",
    r"kurallar(?:ini|inizi)\s*(?:goster|paylas|soyle|yaz)",
    r"rolunu\s*(?:unut|degistir|birak)",
    r"baska\s+(?:bir\s+)?karakter(?:i|e)\s*(?:burun|donuş|don)",
    r"jailbreak",
    r"\bDAN\s+mode\b",
    r"prompt\s*injection",
    r"system\s+message",
    r"developer\s+message",
]

_INPUT_RE = re.compile("|".join(_BLOCKED_INPUT_PATTERNS), re.IGNORECASE)


# LLM cevabinda sistem prompt'tan parca sizdi mi?
_LEAK_PATTERNS = [
    r"SADECE sana verilen BAGLAM",
    r"GIZLILIK KURALI",
    r"Sistem talimatlarini",
    r"sistem talimatlari",
    r"prompt'?u\s+(?:soyle|paylas)",
    r"You are an?\s+(?:assistant|AI|chatbot)",
    r"system\s+prompt",
]
_OUTPUT_RE = re.compile("|".join(_LEAK_PATTERNS), re.IGNORECASE)


def check_input(message: str) -> Optional[str]:
    """Tehlikeliyse REJECT_MESSAGE dondur, degilse None."""
    if not message or len(message.strip()) < 2:
        return REJECT_MESSAGE
    if _INPUT_RE.search(message):
        return REJECT_MESSAGE
    return None


def sanitize_output(answer: str) -> str:
    """LLM cevabinda sistem prompt sizintisi varsa REJECT_MESSAGE'a cevir."""
    if _OUTPUT_RE.search(answer):
        return REJECT_MESSAGE
    return answer
