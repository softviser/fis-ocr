"""Kuyruk çekirdeği testi: JobStore + WorkerPool, FastAPI ve Ollama gerektirmez.

    py -3 tests/test_queue_core.py
"""
import os
import sys
import tempfile
import time

os.environ["RECEIPT2JSON_FAKE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from receipt2json.service.store import JobStore  # noqa: E402
from receipt2json.service.worker import WorkerPool  # noqa: E402


def wait_status(store, job_id, target, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job["status"] == target:
            return job
        if job["status"] == "error":
            raise AssertionError(f"İş hata verdi: {job['error']}")
        time.sleep(0.05)
    raise AssertionError(f"{job_id} '{target}' olmadı: {store.get(job_id)['status']}")


def main():
    tmp = tempfile.mkdtemp(prefix="r2j_core_")
    db = os.path.join(tmp, "jobs.db")
    img = os.path.join(tmp, "fis.png")
    with open(img, "wb") as f:
        f.write(b"\x89PNG fake")

    # --- kuyruk sırası: worker başlamadan 3 iş ekle
    store = JobStore(db)
    ids = [store.create(img, f"fis{i}.png", "two_stage", None) for i in range(3)]
    assert store.queue_position(ids[0]) == 1
    assert store.queue_position(ids[2]) == 3

    # --- worker başlat: bekleyen işleri devralıp bitirmeli (crash-recovery yolu)
    pool = WorkerPool(store, count=1)
    pool.start()
    for jid in ids:
        job = wait_status(store, jid, "done")
        assert job["result"] is not None

    # --- yeni iş submit ile
    jid = store.create(img, "yeni.png", "two_stage", None)
    pool.submit(jid)
    job = wait_status(store, jid, "done")
    import json
    result = json.loads(job["result"])
    assert result["_tax_check"]["valid"] is True, result["_tax_check"]
    assert result["grand_total"] == 100

    # --- bitmiş işin queue_position'ı None
    assert store.queue_position(jid) is None

    pool.stop()

    # --- otomatik temizlik: eski işler silinmeli, görsel dosyası dahil
    old_img = os.path.join(tmp, "eski.png")
    with open(old_img, "wb") as f:
        f.write(b"eski")
    old_id = store.create(old_img, "eski.png", "two_stage", None)
    store.mark_done(old_id, {"grand_total": 1})
    # created_at'i 40 gün geriye çek
    import sqlite3
    with sqlite3.connect(db) as c:
        c.execute("UPDATE jobs SET created_at = created_at - 40*86400 WHERE id=?",
                  (old_id,))
    assert store.purge_older_than(30) == 1
    assert store.get(old_id) is None
    assert not os.path.exists(old_img)
    # yeni işler silinmemeli
    assert store.purge_older_than(30) == 0
    assert store.get(jid) is not None

    print("Kuyruk çekirdeği testleri geçti.")


if __name__ == "__main__":
    main()
