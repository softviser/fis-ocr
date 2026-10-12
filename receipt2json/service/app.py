"""Fiş -> JSON asenkron web servisi.

Çalıştırma:
    uvicorn receipt2json.service.app:app --host 0.0.0.0 --port 8080

Ortam değişkenleri:
    OLLAMA_HOST        Ollama adresi (örn. http://netviser:11434)
    RECEIPT2JSON_DATA  Veri klasörü (varsayılan: ./service_data)
    WORKER_COUNT       Paralel worker sayısı (varsayılan: 1)
    RECEIPT2JSON_FAKE  1 => Ollama'sız sahte çıktı (geliştirme/test)

Akış:
    POST /api/jobs            -> görseli yükle, anında {job_id, queue_position} döner
    GET  /api/jobs/{job_id}   -> durum; bitince result alanında JSON
    GET  /api/jobs            -> son işler
    GET  /health              -> servis + kuyruk durumu
"""
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from ..config import CFG, data_dir, max_upload_bytes
from .store import JobStore
from .worker import RetentionCleaner, WorkerPool

DATA_DIR = data_dir()
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "jobs.db")

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_UPLOAD_BYTES = max_upload_bytes()
MAX_UPLOAD_MB = CFG["max_upload_mb"]

store: JobStore | None = None
workers: WorkerPool | None = None
cleaner: RetentionCleaner | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, workers, cleaner
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    store = JobStore(DB_PATH)
    workers = WorkerPool(store)
    workers.start()
    cleaner = RetentionCleaner(store)
    cleaner.start()
    yield
    cleaner.stop()
    workers.stop()


app = FastAPI(title="Fiş OCR Servisi", version="1.0", lifespan=lifespan)

_HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_PATH = os.path.join(_HERE, "upload_page.html")
ARCHIVE_PATH = os.path.join(_HERE, "archive_page.html")
DOCS_PATH = os.path.join(_HERE, "api_docs.html")


def _page(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
def index():
    return _page(PAGE_PATH)


@app.get("/arsiv", response_class=HTMLResponse)
def archive():
    return _page(ARCHIVE_PATH)


# FastAPI'nin kendi /docs'u (Swagger) kalsın; el yazımı doküman /api-docs'ta
@app.get("/api-docs", response_class=HTMLResponse)
def api_docs():
    return _page(DOCS_PATH)


def _public_job(job: dict) -> dict:
    out = {
        "job_id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "original_filename": job["orig_name"],
        "mode": job["mode"],
    }
    if job["status"] == "queued":
        out["queue_position"] = store.queue_position(job["id"])
    if job["status"] == "processing":
        out["started_at"] = job["started_at"]
    if job["status"] == "done":
        out["finished_at"] = job["finished_at"]
        out["elapsed_seconds"] = (
            round(job["finished_at"] - (job["started_at"] or job["created_at"]), 1)
        )
        out["result"] = json.loads(job["result"])
    if job["status"] == "error":
        out["finished_at"] = job["finished_at"]
        out["error"] = job["error"]
    return out


@app.post("/api/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    mode: str = Form("two_stage"),
    ocr_model: str | None = Form(None),
):
    if mode not in ("two_stage", "direct"):
        raise HTTPException(422, "mode 'two_stage' veya 'direct' olmalı")
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(415, f"Desteklenmeyen dosya türü: {ext or 'bilinmiyor'}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Dosya {MAX_UPLOAD_MB} MB sınırını aşıyor")
    if not content:
        raise HTTPException(422, "Boş dosya")

    dest = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}")
    with open(dest, "wb") as f:
        f.write(content)

    job_id = store.create(dest, file.filename, mode, ocr_model)
    workers.submit(job_id)
    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": store.queue_position(job_id),
        "status_url": f"/api/jobs/{job_id}",
    }


@app.post("/api/jobs/base64", status_code=202)
async def create_job_base64(payload: dict):
    """Programatik kullanım: {"data": "<base64 veya data-URL>", "filename": "fis.jpg",
    "mode": "two_stage", "ocr_model": null}"""
    from .stdlib_app import decode_base64_payload

    filename, content, err = decode_base64_payload(payload)
    if err:
        raise HTTPException(422, err)
    mode = payload.get("mode") or "two_stage"
    if mode not in ("two_stage", "direct"):
        raise HTTPException(422, "mode 'two_stage' veya 'direct' olmalı")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(415, f"Desteklenmeyen dosya türü: {ext}")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Dosya {MAX_UPLOAD_MB} MB sınırını aşıyor")

    dest = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}")
    with open(dest, "wb") as f:
        f.write(content)
    job_id = store.create(dest, filename, mode, payload.get("ocr_model") or None)
    workers.submit(job_id)
    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": store.queue_position(job_id),
        "status_url": f"/api/jobs/{job_id}",
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "İş bulunamadı")
    return _public_job(job)


@app.get("/api/jobs/{job_id}/image")
def get_job_image(job_id: str):
    from fastapi.responses import FileResponse

    img = store.image_path(job_id)
    if not img or not os.path.isfile(img):
        raise HTTPException(404, "Görsel bulunamadı")
    return FileResponse(img)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """İş kaydını ve yüklenen görseli sil."""
    if not store.delete(job_id):
        raise HTTPException(404, "İş bulunamadı")
    return {"deleted": job_id}


@app.get("/api/jobs")
def list_jobs(limit: int = 50, status: str | None = None, q: str | None = None):
    return {"jobs": store.recent(min(limit, 200), status=status, q=q)}


@app.get("/api/stats")
def get_stats():
    return store.stats()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "queue_size": workers.queue.qsize() if workers else None,
        "worker_count": workers.count if workers else 0,
        "max_upload_mb": MAX_UPLOAD_MB,
        "retention_days": CFG.get("retention_days") or 0,
        "fake_mode": os.environ.get("RECEIPT2JSON_FAKE") == "1",
    }
