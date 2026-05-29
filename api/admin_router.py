"""Admin endpoint'leri.

Login disinda hepsi JWT bearer ister.
Path prefix: /admin
"""
from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from core.auth import authenticate, create_token, require_admin
from core.budget import token_budget
from core.cache import qa_cache
from core.config import (
    DATA_DIR,
    MAX_UPLOAD_SIZE_MB,
    SERVICE_NAME,
    SERVICE_VERSION,
    get_system_prompt,
    set_system_prompt,
)
from core.session_store import session_store
from core.stats import stats
from core.vector_engine import build_or_update

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


# --- Schemas ---
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class LoginResponse(BaseModel):
    token: str
    expires_in_days: int


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=20, max_length=10000)


class PromptResponse(BaseModel):
    prompt: str


class DocumentInfo(BaseModel):
    name: str
    size_bytes: int
    size_mb: float
    pdf_url: str


class DocumentList(BaseModel):
    documents: List[DocumentInfo]
    total: int


class ReindexResponse(BaseModel):
    added: List[str]
    changed: List[str]
    removed: List[str]
    total: int


class DeleteResponse(BaseModel):
    deleted: str
    reindexed: bool


class StatsResponse(BaseModel):
    service: Dict
    requests: Dict
    cache: Dict
    sessions: Dict
    budget: Dict


# --- Endpoints ---
@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    if not authenticate(req.username, req.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanici adi veya sifre hatali",
        )
    return LoginResponse(token=create_token(req.username), expires_in_days=7)


@router.get("/system-prompt", response_model=PromptResponse)
async def get_prompt(_: str = Depends(require_admin)):
    return PromptResponse(prompt=get_system_prompt())


@router.put("/system-prompt", response_model=PromptResponse)
async def update_prompt(req: PromptRequest, _: str = Depends(require_admin)):
    set_system_prompt(req.prompt)
    log.info("[admin] sistem prompt guncellendi")
    return PromptResponse(prompt=get_system_prompt())


@router.get("/documents", response_model=DocumentList)
async def list_documents(_: str = Depends(require_admin)):
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    items = []
    for p in pdfs:
        size = p.stat().st_size
        items.append(
            DocumentInfo(
                name=p.name,
                size_bytes=size,
                size_mb=round(size / (1024 * 1024), 2),
                pdf_url=f"/pdf/{p.name}",
            )
        )
    return DocumentList(documents=items, total=len(items))


@router.post("/documents/upload", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...), _: str = Depends(require_admin)
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Sadece PDF dosyalari kabul edilir")

    safe_name = Path(file.filename).name
    target = DATA_DIR / safe_name

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    written = 0

    with open(target, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                f.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, f"Dosya {MAX_UPLOAD_SIZE_MB} MB'dan buyuk olamaz")
            f.write(chunk)

    log.info(f"[admin] PDF yuklendi: {safe_name} ({written} bytes)")
    return DocumentInfo(
        name=safe_name,
        size_bytes=written,
        size_mb=round(written / (1024 * 1024), 2),
        pdf_url=f"/pdf/{safe_name}",
    )


@router.delete("/documents/{name}", response_model=DeleteResponse)
async def delete_document(name: str, _: str = Depends(require_admin)):
    safe_name = Path(name).name
    target = DATA_DIR / safe_name
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "Dosya bulunamadi")
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(400, "Sadece PDF silinebilir")

    target.unlink()
    log.info(f"[admin] PDF silindi: {safe_name}")

    # Otomatik reindex (incremental — silinen PDF'in chunk'lari Chroma'dan da gider)
    try:
        await asyncio.get_event_loop().run_in_executor(None, build_or_update, False)
        reindexed = True
    except Exception as e:
        log.exception(f"[admin] silme sonrasi reindex hatasi: {e}")
        reindexed = False

    qa_cache.clear()
    return DeleteResponse(deleted=safe_name, reindexed=reindexed)


@router.post("/reindex", response_model=ReindexResponse)
async def reindex(force: bool = False, _: str = Depends(require_admin)):
    """Pipeline'i tetikle. force=True ise tam rebuild."""
    try:
        report = await asyncio.get_event_loop().run_in_executor(
            None, build_or_update, force
        )
        qa_cache.clear()
        log.info(f"[admin] reindex tamam: {report}")
        return ReindexResponse(**report)
    except Exception as e:
        log.exception(f"[admin] reindex hatasi: {e}")
        raise HTTPException(500, f"Reindex hatasi: {e}")


@router.post("/cache/clear")
async def clear_cache(_: str = Depends(require_admin)):
    qa_cache.clear()
    return {"status": "ok", "message": "Cache temizlendi"}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, _: str = Depends(require_admin)):
    ok = session_store.delete(session_id)
    if not ok:
        raise HTTPException(404, "Session bulunamadi")
    return {"status": "ok", "deleted": session_id}


@router.get("/stats", response_model=StatsResponse)
async def get_stats(_: str = Depends(require_admin)):
    return StatsResponse(
        service={"name": SERVICE_NAME, "version": SERVICE_VERSION},
        requests=stats.stats(),
        cache=qa_cache.stats(),
        sessions=session_store.stats(),
        budget=token_budget.stats(),
    )
