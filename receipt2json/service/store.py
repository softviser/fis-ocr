"""SQLite tabanlı iş deposu — servis yeniden başlasa da işler ve sonuçlar kalır."""
import json
import os
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,          -- queued | processing | done | error
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    image_path  TEXT NOT NULL,
    orig_name   TEXT,
    mode        TEXT NOT NULL DEFAULT 'two_stage',
    ocr_model   TEXT,
    result      TEXT,                   -- JSON string
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


class JobStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------ yazma

    def create(self, image_path: str, orig_name: str | None,
               mode: str, ocr_model: str | None) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO jobs (id, status, created_at, image_path, orig_name, mode, ocr_model)"
                " VALUES (?, 'queued', ?, ?, ?, ?, ?)",
                (job_id, time.time(), image_path, orig_name, mode, ocr_model),
            )
        return job_id

    def mark_processing(self, job_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE jobs SET status='processing', started_at=? WHERE id=?",
                (time.time(), job_id),
            )

    def mark_done(self, job_id: str, result: dict) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE jobs SET status='done', finished_at=?, result=? WHERE id=?",
                (time.time(), json.dumps(result, ensure_ascii=False), job_id),
            )

    def mark_error(self, job_id: str, error: str) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE jobs SET status='error', finished_at=?, error=? WHERE id=?",
                (time.time(), error, job_id),
            )

    def requeue_stale(self) -> list[str]:
        """Servis çökmüşse yarım kalan işleri tekrar kuyruğa almak için."""
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT id FROM jobs WHERE status IN ('queued','processing')"
                " ORDER BY created_at"
            ).fetchall()
            c.execute("UPDATE jobs SET status='queued' WHERE status='processing'")
        return [r["id"] for r in rows]

    # ------------------------------------------------------------ okuma

    def get(self, job_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def queue_position(self, job_id: str) -> int | None:
        """1 tabanlı kuyruk sırası (yalnızca 'queued' işler için)."""
        with self._conn() as c:
            row = c.execute("SELECT created_at, status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["status"] != "queued":
                return None
            ahead = c.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status='queued' AND created_at < ?",
                (row["created_at"],),
            ).fetchone()["n"]
            processing = c.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status='processing'"
            ).fetchone()["n"]
        return ahead + processing + 1

    def recent(self, limit: int = 50, status: str | None = None,
               q: str | None = None) -> list[dict]:
        """Arşiv listesi: isteğe bağlı durum filtresi ve metin araması.

        'done' işlerde sonuçtan kompakt özet (firma, toplam, tarih, şüpheli)
        çıkarılıp 'summary' alanına eklenir.
        """
        sql = ("SELECT id, status, created_at, started_at, finished_at,"
               " orig_name, mode, result FROM jobs")
        conds, params = [], []
        if status:
            conds.append("status = ?")
            params.append(status)
        if q:
            conds.append("(orig_name LIKE ? OR result LIKE ?)")
            params += [f"%{q}%", f"%{q}%"]
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()

        out = []
        for r in rows:
            row = {k: r[k] for k in
                   ("id", "status", "created_at", "finished_at", "orig_name", "mode")}
            row["elapsed_seconds"] = (
                round(r["finished_at"] - r["started_at"], 1)
                if r["status"] == "done" and r["started_at"] and r["finished_at"]
                else None
            )
            row["summary"] = _summarize(r["result"]) if r["result"] else None
            out.append(row)
        return out

    def stats(self) -> dict:
        """Analiz süresi istatistikleri (yalnız tamamlanan işler) + durum sayıları."""
        with self._conn() as c:
            dur = c.execute(
                "SELECT COUNT(*) AS n,"
                " AVG(finished_at - started_at) AS avg_s,"
                " MIN(finished_at - started_at) AS min_s,"
                " MAX(finished_at - started_at) AS max_s"
                " FROM jobs WHERE status='done' AND started_at IS NOT NULL"
                " AND finished_at IS NOT NULL"
            ).fetchone()
            last10 = c.execute(
                "SELECT AVG(s) FROM (SELECT finished_at - started_at AS s FROM jobs"
                " WHERE status='done' AND started_at IS NOT NULL"
                " AND finished_at IS NOT NULL"
                " ORDER BY finished_at DESC LIMIT 10)"
            ).fetchone()[0]
            by_status = dict(c.execute(
                "SELECT status, COUNT(*) FROM jobs GROUP BY status"
            ).fetchall())
        r1 = lambda v: round(v, 1) if v is not None else None  # noqa: E731
        return {
            "done_count": dur["n"],
            "avg_seconds": r1(dur["avg_s"]),
            "avg_seconds_last10": r1(last10),
            "min_seconds": r1(dur["min_s"]),
            "max_seconds": r1(dur["max_s"]),
            "by_status": by_status,
        }

    def image_path(self, job_id: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT image_path FROM jobs WHERE id=?", (job_id,)).fetchone()
        return row["image_path"] if row else None

    def purge_older_than(self, days: float) -> int:
        """Belirtilen günden eski, bitmiş (done/error) işleri ve görsellerini sil.

        Kuyruktaki/işlenen işlere dokunmaz. Silinen kayıt sayısını döndürür.
        """
        cutoff = time.time() - days * 86400
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT id, image_path FROM jobs"
                " WHERE created_at < ? AND status IN ('done','error')",
                (cutoff,),
            ).fetchall()
            if rows:
                c.executemany(
                    "DELETE FROM jobs WHERE id=?", [(r["id"],) for r in rows]
                )
        for r in rows:
            img = r["image_path"]
            if img and os.path.isfile(img):
                try:
                    os.remove(img)
                except OSError:
                    pass
        return len(rows)

    def delete(self, job_id: str) -> bool:
        """İş kaydını ve yüklenen görsel dosyasını sil."""
        with self._lock, self._conn() as c:
            row = c.execute("SELECT image_path FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return False
            c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        img = row["image_path"]
        if img and os.path.isfile(img):
            try:
                os.remove(img)
            except OSError:
                pass  # dosya silinemese de kayıt silindi
        return True


def _summarize(result_json: str) -> dict | None:
    try:
        doc = json.loads(result_json)
    except ValueError:
        return None
    if doc.get("_multi"):
        receipts = doc.get("receipts") or []
        return {
            "multi": True,
            "count": len(receipts),
            "vendor": next((r.get("vendor_name") for r in receipts
                            if r.get("vendor_name")), None),
            "total": sum(r.get("grand_total") or 0 for r in receipts) or None,
            "date": next((r.get("doc_date") for r in receipts
                          if r.get("doc_date")), None),
            "suspect": any(r.get("_suspect") for r in receipts),
            "warnings": sum(len(r.get("_warnings") or []) for r in receipts),
        }
    return {
        "multi": False,
        "count": 1,
        "vendor": doc.get("vendor_name"),
        "total": doc.get("grand_total"),
        "date": doc.get("doc_date"),
        "suspect": bool(doc.get("_suspect")),
        "warnings": len(doc.get("_warnings") or []),
    }
