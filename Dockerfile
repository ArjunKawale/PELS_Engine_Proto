# ─────────────────────────────────────────────
# Base image — slim Python, pinned for stability
# ─────────────────────────────────────────────
FROM python:3.10-slim

# Keeps Python from buffering stdout/stderr (good for logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9000 \
    # Tell HuggingFace/sentence-transformers where to cache models
    TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models

WORKDIR /app

# ─────────────────────────────────────────────
# System dependencies
# ChromaDB needs sqlite3 ≥ 3.35 — the slim image
# ships an older one, so we grab it from backports
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libsqlite3-dev \
    sqlite3 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Python deps — split into two layers so that
# the heavy torch layer is only re-built when
# requirements.txt actually changes
# ─────────────────────────────────────────────
COPY requirements.txt .

# Layer 1 — CPU-only PyTorch (large, changes rarely)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Layer 2 — everything else (your requirements)
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Pre-bake the embedding model into the image
# so the container starts instantly with no
# download at runtime
# ─────────────────────────────────────────────
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('all-MiniLM-L6-v2'); \
print('✅ Model cached successfully')"

# ─────────────────────────────────────────────
# Copy application code LAST — changes here
# won't bust the expensive layers above
# ─────────────────────────────────────────────
COPY . .

# ChromaDB stores its data here — mount a volume
# at this path in production to persist your DB
VOLUME ["/app/chroma_db"]

# Expose for local dev / documentation purposes
EXPOSE $PORT

# ─────────────────────────────────────────────
# Entrypoint — reads $PORT set by Zoho Catalyst
# (or falls back to 9000)
# ─────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-9000}"]