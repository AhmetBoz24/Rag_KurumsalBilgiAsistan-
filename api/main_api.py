"""Kurumsal Bilgi Asistani - FastAPI servisi.

Public endpoints (ogrenci):
  GET  /              -> minimum servis meta
  GET  /health        -> saglik kontrolu
  GET  /documents     -> erisilen PDF listesi
  GET  /pdf/{name}    -> PDF dosyasini servis et (deep link destekli)
  POST /chat          -> tek atisli cevap
  POST /chat/stream   -> SSE streaming
  GET  /history/{sid} -> belirli session'in mesajlari
  POST /session       -> yeni session uret

Admin endpoints: /admin/* (admin_router'a delege edilir)
"""
import json
import logging
import os
import time
import uuid
from typing import AsyncIterator, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from core.budget import token_budget
from core.cache import qa_cache
from core.config import (
    ALLOWED_ORIGINS,
    CHROMA_DIR,
    DATA_DIR,
    FINAL_TOP_K,
    GROQ_MODEL,
    MAX_HISTORY_TURNS,
    MAX_MESSAGE_LEN,
    MAX_SOURCES_RETURNED,
    RATE_LIMIT_PER_DAY,
    RATE_LIMIT_PER_HOUR,
    RATE_LIMIT_PER_MINUTE,
    SERVICE_NAME,
    SERVICE_VERSION,
    get_system_prompt,
)
from core.guard import check_input, sanitize_output, REJECT_MESSAGE
from core.rate_limiter import limiter, session_limiter
from core.retriever import HybridRetriever
from core.session_store import session_store
from core.stats import stats


# --- Logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("kurumsal-bilgi")


# --- Ortam ---
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    log.warning("GROQ_API_KEY .env'de yok!")


# --- App ---
app = FastAPI(
    title=SERVICE_NAME,
    description="Istanbul Gedik Universitesi RAG asistani",
    version=SERVICE_VERSION,
    docs_url=None,        # Swagger UI'yi prod'da kapali tut
    redoc_url=None,
    openapi_url=None,     # OpenAPI sema da gizli
)

# Rate limit middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# --- Admin router ---
from api.admin_router import router as admin_router  # noqa: E402

app.include_router(admin_router)


# --- Servis state ---
log.info(f"Embedding modeli yukleniyor: {os.getenv('EMBEDDING_MODEL', 'default')}")
retriever: Optional[HybridRetriever] = None
llm: Optional[ChatGroq] = None

try:
    retriever = HybridRetriever()
    log.info("Hybrid retriever hazir.")
except Exception as e:
    log.exception(f"Retriever yuklenemedi: {e}")

try:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    log.info(f"Groq LLM hazir: {GROQ_MODEL}")
except Exception as e:
    log.exception(f"Groq baglanamadi: {e}")


# --- Schemas ---
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LEN)
    session_id: Optional[str] = None
    top_k: Optional[int] = Field(default=FINAL_TOP_K, ge=1, le=10)


class SourceItem(BaseModel):
    source: str
    page: int
    madde: Optional[str] = None
    bolum: Optional[str] = None
    snippet: str
    pdf_url: str
    deep_link: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[SourceItem]
    latency_ms: int
    cached: bool = False


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict]


class DocumentPublic(BaseModel):
    name: str
    pages: Optional[int] = None
    pdf_url: str


class DocumentsResponse(BaseModel):
    documents: List[DocumentPublic]
    total: int


# --- Helpers ---
def make_pdf_url(filename: str, page: Optional[int] = None) -> str:
    base = f"/pdf/{filename}"
    return f"{base}#page={page}" if page else base


def to_source_items(docs: List[tuple]) -> List[SourceItem]:
    out: List[SourceItem] = []
    for text, md in docs:
        src = md.get("source", "")
        page = int(md.get("page", 1))
        snippet = text[:240] + ("..." if len(text) > 240 else "")
        out.append(
            SourceItem(
                source=src,
                page=page,
                madde=md.get("madde"),
                bolum=md.get("bolum"),
                snippet=snippet,
                pdf_url=f"/pdf/{src}",
                deep_link=f"/pdf/{src}#page={page}",
            )
        )
    return out


# Selamlama, tesekkur, aciklama isteme veya konu disi reddi gibi
# cevaplarda kaynaklar alakasiz oluyor. Bu helper bunlari yakalar.
_LOW_INFO_TRIGGERS = [
    "merhaba",
    "rica ederim",
    "iyi gunler",
    "size nasil yardimci",
    "size daha dogru bilgi verebilmem",
    "hangi bolum",
    "hangi sinif",
    "hangi fakulte",
    "belirtir misiniz",
    "ben gedik universitesi'nin kurumsal",
    "kurumsal bilgi asistaniyim",
    "baska bir sorunuz olursa",
]


def _normalize_tr(s: str) -> str:
    """Karsilastirmada Turkce karakterleri normalize et."""
    return (
        s.lower()
        .replace("ı", "i").replace("İ", "i")
        .replace("ş", "s").replace("ç", "c")
        .replace("ğ", "g").replace("ü", "u")
        .replace("ö", "o")
    )


def is_low_info_answer(answer: str) -> bool:
    """Cevap selamlama/aciklama isteme/konu disi reddi ise True."""
    if not answer:
        return True
    lower = _normalize_tr(answer.strip())
    if any(t in lower for t in _LOW_INFO_TRIGGERS) and len(answer.split()) < 45:
        return True
    return False


def filter_sources_for_output(
    items: List[SourceItem], answer: str
) -> List[SourceItem]:
    """Selam/aciklama ise bos liste; aksi halde en alakali MAX_SOURCES_RETURNED tane."""
    if is_low_info_answer(answer):
        return []
    return items[:MAX_SOURCES_RETURNED]


def format_context(docs: List[tuple]) -> str:
    blocks = []
    for i, (text, md) in enumerate(docs, 1):
        bits = []
        if md.get("madde"):
            bits.append(md["madde"])
        if md.get("bolum"):
            bits.append(md["bolum"])
        if md.get("source"):
            bits.append(f"{md['source']} s.{md.get('page', '?')}")
        header = " | ".join(bits) if bits else f"Kaynak {i}"
        blocks.append(f"[{header}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, history: List[Dict], context: str) -> List:
    msgs = [SystemMessage(content=get_system_prompt())]
    for turn in history[-MAX_HISTORY_TURNS * 2 :]:
        if turn["role"] == "user":
            msgs.append(HumanMessage(content=turn["content"]))
        else:
            msgs.append(AIMessage(content=turn["content"]))
    msgs.append(
        HumanMessage(content=f"BAGLAM:\n{context}\n\nSORU:\n{question}\n\nCEVAP:")
    )
    return msgs


def estimate_tokens(question: str, context: str, history: List[Dict]) -> int:
    """Kabaca: 4 char ~ 1 token."""
    total = len(question) + len(context) + len(get_system_prompt())
    total += sum(len(m["content"]) for m in history)
    return total // 4


def precheck_request(req: ChatRequest) -> tuple[str, Optional[JSONResponse]]:
    """Inputu temizle, gizli engelleyici cevap varsa donder.

    Returns: (session_id, early_response_or_None)
    """
    sid = session_store.get_or_create(req.session_id)

    # Input guard
    blocked = check_input(req.message)
    if blocked:
        stats.record_rejected_input()
        early = JSONResponse(
            {
                "answer": blocked,
                "session_id": sid,
                "sources": [],
                "latency_ms": 0,
                "cached": False,
            }
        )
        return sid, early

    # Session rate
    if not session_limiter.check(sid):
        stats.record_rejected_rate()
        early = JSONResponse(
            status_code=429,
            content={"detail": "Cok sik istek gonderiyorsunuz, lutfen biraz bekleyin."},
        )
        return sid, early

    return sid, None


# --- Endpoints ---
@app.get("/")
async def root():
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "status": "online",
    }


@app.get("/health")
async def health():
    chroma_ok = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())
    doc_count = None
    if retriever is not None:
        try:
            doc_count = retriever.vectorstore._collection.count()
        except Exception:
            pass
    return {
        "status": "ok" if (retriever and llm) else "degraded",
        "retriever_ready": retriever is not None,
        "llm_ready": llm is not None,
        "chroma_persisted": chroma_ok,
        "indexed_chunks": doc_count,
    }


@app.get("/documents", response_model=DocumentsResponse)
async def public_documents():
    """Ogrencinin gorebilecegi PDF listesi."""
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    items = []
    for p in pdfs:
        items.append(
            DocumentPublic(
                name=p.name,
                pdf_url=f"/pdf/{p.name}",
                pages=None,  # PDF acmadan sayfa sayisi bilinmiyor, opsiyonel
            )
        )
    return DocumentsResponse(documents=items, total=len(items))


@app.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    """PDF dosyasini servis eder. Frontend iframe / pdf viewer kullanacak.

    Browser native deep link: /pdf/x.pdf#page=5 (frontend href olarak verir).
    """
    safe_name = os.path.basename(filename)
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(400, "Sadece PDF servis edilir")
    path = DATA_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Dosya bulunamadi")
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
    )


@app.post("/session")
async def new_session():
    sid = session_store.new_session()
    return {"session_id": sid}


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    msgs = session_store.get_history(session_id)
    return HistoryResponse(session_id=session_id, messages=msgs)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT_PER_MINUTE)
@limiter.limit(RATE_LIMIT_PER_HOUR)
@limiter.limit(RATE_LIMIT_PER_DAY)
async def chat(request: Request, req: ChatRequest = Body(...)):
    if retriever is None or llm is None:
        raise HTTPException(503, "Servis baslatilamadi.")

    sid, early = precheck_request(req)
    if early is not None:
        return early

    t0 = time.time()
    history = session_store.get_history(sid)

    # Cache (sadece stateless: ilk mesaj)
    cache_eligible = len(history) == 0
    if cache_eligible:
        cached = qa_cache.get(req.message)
        if cached:
            answer, sources = cached
            session_store.append(sid, "user", req.message)
            session_store.append(sid, "assistant", answer)
            latency = int((time.time() - t0) * 1000)
            stats.record_chat(latency)
            return ChatResponse(
                answer=answer,
                session_id=sid,
                sources=[SourceItem(**s) for s in sources],
                latency_ms=latency,
                cached=True,
            )

    try:
        t_retrieve = time.time()
        docs = retriever.retrieve(req.message, top_k=req.top_k or FINAL_TOP_K)
        retrieval_ms = int((time.time() - t_retrieve) * 1000)
        context = format_context(docs)

        est = estimate_tokens(req.message, context, history)
        if not token_budget.can_spend(est + 1000):  # +1000 cevap margin
            stats.record_rejected_budget()
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Bugun icin servis kapasitemiz doldu, lutfen yarin tekrar deneyin."
                },
            )

        messages = build_messages(req.message, history, context)
        t_llm = time.time()
        response = llm.invoke(messages)
        llm_ms = int((time.time() - t_llm) * 1000)
        answer = response.content if hasattr(response, "content") else str(response)
        answer = sanitize_output(answer)

        # Token muhasebesi
        used = token_budget.count_tokens(req.message + context + answer)
        token_budget.add(used)

        session_store.append(sid, "user", req.message)
        session_store.append(sid, "assistant", answer)

        all_source_items = to_source_items(docs)
        visible_sources = filter_sources_for_output(all_source_items, answer)
        sources_serializable = [s.model_dump() for s in visible_sources]

        if cache_eligible and answer != REJECT_MESSAGE:
            qa_cache.set(req.message, answer, sources_serializable)

        latency = int((time.time() - t0) * 1000)
        stats.record_chat(latency)
        log.info(
            f"[chat] sid={sid[:8]} chunks={len(docs)} shown={len(visible_sources)} "
            f"tokens={used} retrieval={retrieval_ms}ms llm={llm_ms}ms total={latency}ms"
        )

        return ChatResponse(
            answer=answer,
            session_id=sid,
            sources=visible_sources,
            latency_ms=latency,
            cached=False,
        )
    except Exception as e:
        err = str(e).lower()
        if "429" in err or "rate_limit" in err:
            return JSONResponse(
                status_code=429,
                content={"detail": "Su an bir yogunluk var, birkac dakika sonra deneyin."},
            )
        stats.record_error()
        log.exception(f"[chat] hata: {e}")
        raise HTTPException(500, "Sistem bir hata ile karsilasti.")


@app.post("/chat/stream")
@limiter.limit(RATE_LIMIT_PER_MINUTE)
@limiter.limit(RATE_LIMIT_PER_HOUR)
@limiter.limit(RATE_LIMIT_PER_DAY)
async def chat_stream(request: Request, req: ChatRequest = Body(...)):
    if retriever is None or llm is None:
        raise HTTPException(503, "Servis baslatilamadi.")

    sid, early = precheck_request(req)
    if early is not None:
        return early

    async def event_stream() -> AsyncIterator[str]:
        t0 = time.time()
        try:
            history = session_store.get_history(sid)
            t_retrieve = time.time()
            docs = retriever.retrieve(req.message, top_k=req.top_k or FINAL_TOP_K)
            retrieval_ms = int((time.time() - t_retrieve) * 1000)
            context = format_context(docs)

            est = estimate_tokens(req.message, context, history)
            if not token_budget.can_spend(est + 1000):
                stats.record_rejected_budget()
                yield (
                    "event: error\ndata: "
                    + json.dumps(
                        {
                            "message": "Bugun icin servis kapasitemiz doldu, lutfen yarin tekrar deneyin."
                        }
                    )
                    + "\n\n"
                )
                return

            messages = build_messages(req.message, history, context)

            yield f"event: meta\ndata: {json.dumps({'session_id': sid})}\n\n"

            parts: List[str] = []
            async for chunk in llm.astream(messages):
                piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                if not piece:
                    continue
                parts.append(piece)
                yield f"event: token\ndata: {json.dumps({'text': piece})}\n\n"

            full = sanitize_output("".join(parts))

            # Sanitize sonrasi tum metin degistiyse uyari gonder
            if full != "".join(parts):
                yield f"event: redacted\ndata: {json.dumps({'text': full})}\n\n"

            used = token_budget.count_tokens(req.message + context + full)
            token_budget.add(used)

            session_store.append(sid, "user", req.message)
            session_store.append(sid, "assistant", full)

            all_source_items = to_source_items(docs)
            visible_sources = filter_sources_for_output(all_source_items, full)
            payload = {
                "sources": [s.model_dump() for s in visible_sources],
                "latency_ms": int((time.time() - t0) * 1000),
            }
            yield f"event: sources\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

            stats.record_stream(payload["latency_ms"])
            llm_ms = payload["latency_ms"] - retrieval_ms
            log.info(
                f"[stream] sid={sid[:8]} chunks={len(docs)} shown={len(visible_sources)} "
                f"tokens={used} retrieval={retrieval_ms}ms llm={llm_ms}ms "
                f"total={payload['latency_ms']}ms"
            )
        except Exception as e:
            err = str(e).lower()
            msg = (
                "Su an bir yogunluk var, birkac dakika sonra deneyin."
                if ("429" in err or "rate_limit" in err)
                else "Sistem bir hata ile karsilasti."
            )
            stats.record_error()
            log.exception(f"[stream] hata: {e}")
            yield f"event: error\ndata: {json.dumps({'message': msg})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
