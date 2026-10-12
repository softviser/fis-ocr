"""Web servis uçtan uca testi (Ollama gerektirmez; FAKE mod).

    set RECEIPT2JSON_FAKE=1 && py -3 tests/test_service.py
"""
import io
import os
import sys
import tempfile
import time

os.environ["RECEIPT2JSON_FAKE"] = "1"
os.environ["RECEIPT2JSON_DATA"] = tempfile.mkdtemp(prefix="r2j_test_")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from receipt2json.service.app import app  # noqa: E402

# 1x1 PNG
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd40000000049454e44ae426082"
)


def main():
    with TestClient(app) as client:
        # health
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["fake_mode"], r.text

        # geçersiz dosya türü reddedilmeli
        r = client.post("/api/jobs", files={"file": ("x.txt", io.BytesIO(b"abc"), "text/plain")})
        assert r.status_code == 415, r.text

        # iş gönder
        r = client.post("/api/jobs", files={"file": ("fis.png", io.BytesIO(PNG), "image/png")})
        assert r.status_code == 202, r.text
        body = r.json()
        job_id = body["job_id"]
        assert body["status"] == "queued" and body["queue_position"] >= 1

        # sonuç gelene kadar bekle
        for _ in range(50):
            r = client.get(f"/api/jobs/{job_id}")
            assert r.status_code == 200
            j = r.json()
            if j["status"] == "done":
                break
            assert j["status"] in ("queued", "processing"), j
            time.sleep(0.1)
        else:
            raise AssertionError(f"İş zamanında bitmedi: {j}")

        result = j["result"]
        assert result["_tax_check"]["valid"] is True
        assert result["grand_total"] == 100
        assert "elapsed_seconds" in j

        # olmayan iş 404
        assert client.get("/api/jobs/yok12345").status_code == 404

        # listeleme
        jobs = client.get("/api/jobs").json()["jobs"]
        assert any(x["id"] == job_id for x in jobs)

    print("Servis testleri geçti.")


if __name__ == "__main__":
    main()
