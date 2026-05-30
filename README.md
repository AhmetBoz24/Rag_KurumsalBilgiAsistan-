---
title: Gedik Kurumsal Bilgi Asistani
emoji: 🎓
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Istanbul Gedik Universitesi yonetmelik RAG asistani
---

# Kurumsal Bilgi Asistani — Gedik Universitesi RAG

Universite yonetmelikleri, staj yonergeleri ve SSS metinleri uzerinde
dogal dilde sorgu yapilabilen, ogrenci + admin panelli RAG asistani.

## Ozellikler

- **Hibrit Retrieval**: turkish-e5-large (dense) + BM25 (sparse) + RRF + BGE-Reranker
- **Akilli Chunking**: MADDE/BOLUM duyarli, metadata + kaynak gosterimli
- **Incremental Indexing**: sadece degisen PDF'leri reindex et (hash bazli)
- **Cok katmanli koruma**:
  - IP + session bazli rate limiting
  - Gunluk token butcesi
  - LRU Q&A cache
  - Prompt injection guard (giris ve cikis)
- **Conversation Memory**: 5 turn'luk hafiza, TTL'li session store
- **Streaming**: SSE ile token-token cevap
- **Admin Panel**: JWT auth, PDF yukleme/silme, system prompt edit, stats
- **PDF Deep Link**: kaynaklara tiklanarak ilgili sayfanin acilmasi

## Mimari

```
PDF (data/*.pdf)
   |
   v
pdf_engine.py (PyMuPDF + temizleme)
   |
   v
chunkers/agentic.py (MADDE-aware + metadata)
   |
   v
vector_engine.py (hash-based incremental)
   |
   +-----> Chroma (dense, turkish-e5-large)
   |
   +-----> BM25 (sparse)
   |
   v
HybridRetriever (RRF + BGE-reranker)
   |
   v
api/main_api.py  ────── PUBLIC ──────> ogrenci
                ────── /admin/* ─────> admin (JWT)
```

## Kurulum

```powershell
# 1) Sanal ortam
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Admin sifresi + JWT secret olustur
python scripts/setup_admin.py
# -> Ekrana sifre yazilir, .env'e hash + secret kaydedilir

# 3) .env'de GROQ_API_KEY ekle
# GROQ_API_KEY=gsk_...

# 4) PDF'leri data/ klasorune koy, sonra indeksle
python pipeline.py

# 5) Servisi calistir
python -m api.main_api
# veya: uvicorn api.main_api:app --reload --port 8000
```

## .env Dosyasi (ornek)

```
GROQ_API_KEY=gsk_...
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=<bcrypt-hash>
JWT_SECRET=<uzun-random-string>

# Opsiyonel override'lar
ALLOWED_ORIGINS=http://localhost:3000,https://gedik-asistan.example.com
RATE_LIMIT_PER_MINUTE=10/minute
DAILY_TOKEN_BUDGET=1000000
SESSION_TTL_SECONDS=1800
```

## Endpoint Ozeti

### Public
- `GET /` — servis durumu
- `GET /health` — saglik kontrolu
- `GET /documents` — erisilen PDF listesi
- `GET /pdf/{name}` — PDF dosyasini stream (deep link destekli)
- `POST /session` — yeni session
- `GET /history/{sid}` — session mesaj gecmisi
- `POST /chat` — tek atisli cevap
- `POST /chat/stream` — SSE streaming

### Admin (JWT bearer)
- `POST /admin/login` — JWT token al
- `GET/PUT /admin/system-prompt`
- `GET /admin/documents`, `POST /admin/documents/upload`, `DELETE /admin/documents/{name}`
- `POST /admin/reindex`
- `POST /admin/cache/clear`
- `GET /admin/stats`

Detayli sema icin `FRONTEND_CONTEXT.md` dosyasina bak.

## Pipeline

```powershell
# Incremental — sadece yeni/degisen PDF
python pipeline.py

# Tam rebuild
python pipeline.py --force
```

Admin paneli "Reindex" butonu ile de tetiklenebilir.

## Frontend

Frontend bu projeden ayri repo'da:
`c:\Users\bozah\Desktop\Rag_KurumsalBilgiAsistanı_frontend\`

Frontend'i kurarken `FRONTEND_CONTEXT.md` (API kontrati) ve
`FRONTEND_PROMPT.md` (implementation gorevi) dosyalarini Claude'a ver.

## Test (manuel)

```powershell
# Health
curl http://localhost:8000/health

# Public chat
curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d '{"message":"Ozel ogrenci kabul sartlari?"}'

# Admin login
curl -X POST http://localhost:8000/admin/login `
  -H "Content-Type: application/json" `
  -d '{"username":"admin","password":"<sifre>"}'

# Admin stats
curl http://localhost:8000/admin/stats `
  -H "Authorization: Bearer <token>"
```

## Konfigurasyon

Tum ayarlar `core/config.py` icinde, ortam degiskenleri ile override edilebilir.
Onemli olanlar:

| Degisken | Default | Aciklama |
|---|---|---|
| `EMBEDDING_MODEL` | `ytu-ce-cosmos/turkish-e5-large` | Embedding |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM |
| `FINAL_TOP_K` | 4 | LLM'e giden chunk sayisi |
| `RATE_LIMIT_PER_MINUTE` | 10/minute | IP basina dakikada |
| `DAILY_TOKEN_BUDGET` | 1,000,000 | Gunluk toplam token |
| `SESSION_TTL_SECONDS` | 1800 | Session inaktivite TTL |
| `MAX_HISTORY_TURNS` | 5 | Konusma hafiza (turn = 2 mesaj) |
| `ALLOWED_ORIGINS` | `*` | CORS (virgulle ayrik) |

## Branch

`feature/rag-improvements` — su anda aktif. Test sonrasi `main`'e merge.
