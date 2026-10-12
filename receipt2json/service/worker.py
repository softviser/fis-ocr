"""Arka plan worker'ı: kuyruktan iş çeker, Ollama pipeline'ını çalıştırır.

Ollama GPU'yu tek iş için kullandığından varsayılan 1 worker thread'dir;
WORKER_COUNT ortam değişkeniyle artırılabilir.
"""
import logging
import os
import queue
import threading
import time
import traceback

from ..config import CFG
from ..pipeline import ReceiptPipeline
from .store import JobStore

log = logging.getLogger("receipt2json.worker")

FAKE_MODE = os.environ.get("RECEIPT2JSON_FAKE") == "1"


class RetentionCleaner(threading.Thread):
    """config'teki retention_days'ten eski işleri periyodik olarak siler.

    retention_days 0 veya null ise hiçbir şey yapmaz.
    """

    def __init__(self, store: JobStore):
        super().__init__(name="retention-cleaner", daemon=True)
        self.store = store
        self._stop = threading.Event()

    def run(self) -> None:
        interval = max(0.25, float(CFG.get("cleanup_interval_hours") or 6)) * 3600
        while not self._stop.is_set():
            days = CFG.get("retention_days")
            if days:
                try:
                    n = self.store.purge_older_than(float(days))
                    if n:
                        log.info("Otomatik temizlik: %d günden eski %d iş silindi",
                                 days, n)
                except Exception:  # noqa: BLE001
                    log.exception("Otomatik temizlik hatası")
            self._stop.wait(interval)

    def stop(self) -> None:
        self._stop.set()


def _fake_extract(image_path: str) -> dict:
    """Ollama olmadan uçtan uca test için sahte çıktı (RECEIPT2JSON_FAKE=1)."""
    time.sleep(0.2)
    return {
        "vendor_name": "TEST FİRMA",
        "vendor_tax_no": "3620273655",
        "doc_date": "2026-01-01",
        "doc_no": "0001",
        "category": "yazar_kasa",
        "payment_method": "nakit",
        "currency": "TRY",
        "items": [{"name": "TEST", "qty": 1, "unit_price": 100, "vat_rate": 10, "total": 100}],
        "subtotal": 90.91,
        "vat_rate": 10,
        "vat_amount": 9.09,
        "grand_total": 100,
        "notes": f"fake mode; file={os.path.basename(image_path)}",
    }


class WorkerPool:
    def __init__(self, store: JobStore, host: str | None = None, count: int | None = None):
        self.store = store
        self.queue: queue.Queue[str] = queue.Queue()
        self.pipeline = ReceiptPipeline(host=host)
        self.count = count or int(CFG["worker_count"])
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        # Yarım kalmış / bekleyen işleri devral
        for job_id in self.store.requeue_stale():
            self.queue.put(job_id)
        for i in range(self.count):
            t = threading.Thread(target=self._loop, name=f"worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("WorkerPool başladı: %d thread, %d devralınan iş",
                 self.count, self.queue.qsize())

    def stop(self) -> None:
        self._stop.set()

    def submit(self, job_id: str) -> None:
        self.queue.put(job_id)

    # ------------------------------------------------------------ iç akış

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            self._process(job_id)
            self.queue.task_done()

    def _process(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if not job or job["status"] not in ("queued", "processing"):
            return
        self.store.mark_processing(job_id)
        t0 = time.time()
        try:
            if FAKE_MODE:
                result = self.pipeline.postprocess(_fake_extract(job["image_path"]))
            else:
                pipe = self.pipeline
                if job["ocr_model"]:
                    ocr_text = pipe.ocr(job["image_path"], model=job["ocr_model"])
                    result = pipe.postprocess(pipe.structure(ocr_text), ocr_text)
                else:
                    docs = pipe.run_auto(
                        job["image_path"], mode=job["mode"] or "two_stage"
                    )
                    if len(docs) == 1:
                        result = docs[0]
                    else:
                        result = {"_multi": True, "receipt_count": len(docs),
                                  "receipts": docs}
            self.store.mark_done(job_id, result)
            log.info("İş bitti %s (%.1fs)", job_id, time.time() - t0)
        except Exception:  # noqa: BLE001 - worker asla ölmemeli
            err = traceback.format_exc(limit=5)
            self.store.mark_error(job_id, err)
            log.error("İş hatası %s: %s", job_id, err)
