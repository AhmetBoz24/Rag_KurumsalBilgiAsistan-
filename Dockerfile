# Hugging Face Spaces icin Docker imaji
# CPU-only torch, model pre-download, chroma indeks bake-in

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface \
    PORT=7860

WORKDIR /app

# Sistem bagimliliklari
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# pip + CPU-only torch (cok daha kucuk)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Proje dosyalari
COPY . .

# Modelleri imaja gomelim (ilk istek hizli olsun)
# Note: build sirasinda internet gerek
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('ytu-ce-cosmos/turkish-e5-large')" && \
    python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)" || \
    echo "Reranker indirilemedi, runtime'da indirilecek"

# data/ klasorundeki PDF'leri indeksle (chroma + bm25 imaja gomulur)
RUN python pipeline.py

# HF Spaces default port
EXPOSE 7860

# HF Spaces calistirma izinleri (uvicorn root olmayan kullanici ile calismali)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Uvicorn baslat
CMD ["python", "-m", "uvicorn", "api.main_api:app", "--host", "0.0.0.0", "--port", "7860"]
