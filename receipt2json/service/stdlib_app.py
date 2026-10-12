"""FastAPI'siz, saf stdlib HTTP servisi — aynı API, aynı kuyruk altyapısı.

FastAPI sürümü (app.py) pydantic'in native DLL'ini ister; bazı kilitli Windows
makinelerde (Uygulama Denetimi) bu engellenir. Bu sürüm yalnızca standart
kütüphane kullanır ve her yerde çalışır.

    python -m receipt2json.service.stdlib_app --port 8077
"""
import argparse
import base64
import binascii
import json
import logging
import os
import re
import time
import uuid
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from ..config import CFG, data_dir, max_upload_bytes
from .store import JobStore
from .worker import RetentionCleaner, WorkerPool

DATA_DIR = data_dir()
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "jobs.db")
_HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_PATH = os.path.join(_HERE, "upload_page.html")
ARCHIVE_PATH = os.path.join(_HERE, "archive_page.html")
DOCS_PATH = os.path.join(_HERE, "api_docs.html")

IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".webp": "image/webp", ".bmp": "image/bmp", ".tiff": "image/tiff"}

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_UPLOAD_BYTES = max_upload_bytes()
MAX_UPLOAD_MB = CFG["max_upload_mb"]

log = logging.getLogger("receipt2json.http")

store: JobStore | None = None
workers: WorkerPool | None = None


def public_job(job: dict) -> dict:
    out = {
        "job_id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "orig_name": job["orig_name"],
        "mode": job["mode"],
    }
    if job["status"] == "queued":
        out["queue_position"] = store.queue_position(job["id"])
    if job["status"] == "processing":
        out["started_at"] = job["started_at"]
    if job["status"] == "done":
        out["finished_at"] = job["finished_at"]
        out["elapsed_seconds"] = round(
            job["finished_at"] - (job["started_at"] or job["created_at"]), 1
        )
        out["result"] = json.loads(job["result"])
    if job["status"] == "error":
        out["finished_at"] = job["finished_at"]
        out["error"] = job["error"]
    return out


def create_job(filename: str, content: bytes, mode: str, ocr_model: str | None):
    """Doğrula + kaydet + kuyruğa al. (yanıt_kodu, yanıt_gövdesi) döner."""
    if not content:
        return 422, {"detail": "Görsel verisi boş"}
    if mode not in ("two_stage", "direct"):
        return 422, {"detail": "mode 'two_stage' veya 'direct' olmalı"}
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXTS:
        return 415, {"detail": f"Desteklenmeyen dosya türü: {ext or 'bilinmiyor'}"}
    if len(content) > MAX_UPLOAD_BYTES:
        return 413, {"detail": f"Dosya {MAX_UPLOAD_MB} MB sınırını aşıyor"}

    dest = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}")
    with open(dest, "wb") as f:
        f.write(content)

    job_id = store.create(dest, filename, mode, ocr_model)
    workers.submit(job_id)
    return 202, {
        "job_id": job_id,
        "status": "queued",
        "queue_position": store.queue_position(job_id),
        "status_url": f"/api/jobs/{job_id}",
    }


# data-URL önekini ve base64 gövdesini ayıkla
_DATAURL_RE = re.compile(r"^data:image/(\w+);base64,", re.IGNORECASE)

_MIME_EXT = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp",
             "bmp": ".bmp", "tiff": ".tiff"}


def decode_base64_payload(payload: dict):
    """{"data": "<base64 veya data-URL>", "filename": opsiyonel} -> (ad, bytes)."""
    data = payload.get("data") or payload.get("image") or ""
    if not isinstance(data, str) or not data:
        return None, None, "data alanı (base64) zorunlu"
    filename = payload.get("filename")
    m = _DATAURL_RE.match(data)
    if m:
        data = data[m.end():]
        if not filename:
            filename = "pano" + _MIME_EXT.get(m.group(1).lower(), ".png")
    if not filename:
        filename = "upload.png"
    try:
        content = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None, None, "data alanı geçerli base64 değil"
    return filename, content, None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------ yardımcılar

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, path: str) -> None:
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # noqa: A002
        log.info("%s %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------ GET

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._html(PAGE_PATH)
        if path in ("/arsiv", "/archive"):
            return self._html(ARCHIVE_PATH)
        if path in ("/docs", "/api-docs", "/dokuman"):
            return self._html(DOCS_PATH)
        if path == "/openapi.json":
            with open(os.path.join(_HERE, "openapi.json"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/stats":
            return self._json(200, store.stats())
        if path == "/health":
            return self._json(200, {
                "status": "ok",
                "queue_size": workers.queue.qsize(),
                "worker_count": workers.count,
                "max_upload_mb": MAX_UPLOAD_MB,
                "retention_days": CFG.get("retention_days") or 0,
                "fake_mode": os.environ.get("RECEIPT2JSON_FAKE") == "1",
            })
        m = re.fullmatch(r"/api/jobs/([0-9a-f]+)/image", path)
        if m:
            return self._serve_image(m.group(1))
        m = re.fullmatch(r"/api/jobs/([0-9a-f]+)", path)
        if m:
            job = store.get(m.group(1))
            if not job:
                return self._json(404, {"detail": "İş bulunamadı"})
            return self._json(200, public_job(job))
        if path == "/api/jobs":
            qs = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            limit = min(int((qs.get("limit") or ["50"])[0] or 50), 200)
            status = (qs.get("status") or [None])[0] or None
            q = (qs.get("q") or [None])[0] or None
            return self._json(200, {"jobs": store.recent(limit, status=status, q=q)})
        return self._json(404, {"detail": "Bulunamadı"})

    def _serve_image(self, job_id: str) -> None:
        img = store.image_path(job_id)
        if not img or not os.path.isfile(img):
            return self._json(404, {"detail": "Görsel bulunamadı"})
        ext = os.path.splitext(img)[1].lower()
        with open(img, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", IMAGE_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------ DELETE

    def do_DELETE(self):  # noqa: N802
        path = self.path.split("?")[0]
        m = re.fullmatch(r"/api/jobs/([0-9a-f]+)", path)
        if not m:
            return self._json(404, {"detail": "Bulunamadı"})
        if store.delete(m.group(1)):
            return self._json(200, {"deleted": m.group(1)})
        return self._json(404, {"detail": "İş bulunamadı"})

    # ------------------------------------------------------------ POST

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length") or 0)
        # base64 kodlama ~%33 büyütür; payı buna göre bırak
        if length > MAX_UPLOAD_BYTES * 3 // 2 + 4096:
            return self._json(413, {"detail": f"Dosya {MAX_UPLOAD_MB} MB sınırını aşıyor"})

        if path == "/api/jobs":
            return self._post_multipart(length)
        if path == "/api/jobs/base64":
            return self._post_base64(length)
        return self._json(404, {"detail": "Bulunamadı"})

    def _post_multipart(self, length: int) -> None:
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return self._json(415, {"detail": "multipart/form-data bekleniyor"})

        body = self.rfile.read(length)
        msg = BytesParser(policy=email_default).parsebytes(
            b"Content-Type: " + ctype.encode("latin-1") + b"\r\n\r\n" + body
        )

        filename, content, mode, ocr_model = None, None, "two_stage", None
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "file":
                filename = part.get_filename() or "fis.jpg"
                content = part.get_payload(decode=True)
            elif name == "mode":
                mode = (part.get_payload(decode=True) or b"").decode("utf-8").strip() or "two_stage"
            elif name == "ocr_model":
                ocr_model = (part.get_payload(decode=True) or b"").decode("utf-8").strip() or None

        if content is None:
            return self._json(422, {"detail": "file alanı boş"})
        code, payload = create_job(filename, content, mode, ocr_model)
        self._json(code, payload)

    def _post_base64(self, length: int) -> None:
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._json(422, {"detail": "Geçerli JSON gövde bekleniyor"})
        if not isinstance(payload, dict):
            return self._json(422, {"detail": "JSON nesnesi bekleniyor"})

        filename, content, err = decode_base64_payload(payload)
        if err:
            return self._json(422, {"detail": err})
        code, resp = create_job(
            filename, content,
            payload.get("mode") or "two_stage",
            payload.get("ocr_model") or None,
        )
        self._json(code, resp)


def main() -> None:
    global store, workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    store = JobStore(DB_PATH)
    workers = WorkerPool(store)
    workers.start()
    RetentionCleaner(store).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("Fiş OCR servisi: http://%s:%d", args.host, args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        workers.stop()


if __name__ == "__main__":
    main()
