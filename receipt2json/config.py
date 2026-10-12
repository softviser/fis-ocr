"""Merkezi yapılandırma.

Öncelik sırası (yüksekten düşüğe):
  1. Ortam değişkenleri   (OLLAMA_HOST, WORKER_COUNT, RECEIPT2JSON_DATA)
  2. config.json          (proje kökünde; yolu RECEIPT2JSON_CONFIG ile değiştirilebilir)
  3. Koddaki varsayılanlar (DEFAULTS)

config.json alanları:
  ollama_host            Ollama API adresi
  ocr_model              OCR aşaması modeli
  structure_model        Yapılandırma (metin -> JSON) modeli
  vision_model           Tek aşamalı 'direct' mod modeli
  max_upload_mb          Yükleme boyut sınırı (MB)
  worker_count           Paralel worker sayısı
  data_dir               Yüklenen görseller + iş veritabanı klasörü
                         (göreli yol proje köküne göredir)
  retention_days         Bu kadar günden eski işler otomatik silinir
                         (0 veya null = otomatik silme kapalı)
  cleanup_interval_hours Otomatik silme kontrol aralığı (saat)
"""
import json
import logging
import os

log = logging.getLogger("receipt2json.config")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "ollama_host": "http://127.0.0.1:11434",
    "ocr_model": "deepseek-ocr:latest",
    "structure_model": "qwen3-coder:30b",
    "vision_model": "qwen3-vl:8b",
    "max_upload_mb": 20,
    "worker_count": 1,
    "data_dir": "service_data",
    "retention_days": 30,
    "cleanup_interval_hours": 6,
}


def _load() -> dict:
    cfg = dict(DEFAULTS)

    path = os.environ.get(
        "RECEIPT2JSON_CONFIG", os.path.join(PROJECT_ROOT, "config.json")
    )
    # config.local.json (gitignore'da) config.json'u ezer: gerçek sunucu
    # bilgileri repoya girmeden lokalde tanımlanabilir
    local_path = os.path.splitext(path)[0] + ".local.json"
    for p in (path, local_path):
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            cfg.update({k: v for k, v in data.items() if v is not None})
        except (ValueError, OSError) as e:
            log.error("%s okunamadı (%s); atlanıyor", os.path.basename(p), e)

    # Ortam değişkenleri config dosyasını ezer
    if os.environ.get("OLLAMA_HOST"):
        cfg["ollama_host"] = os.environ["OLLAMA_HOST"]
    if os.environ.get("WORKER_COUNT"):
        try:
            cfg["worker_count"] = int(os.environ["WORKER_COUNT"])
        except ValueError:
            pass
    if os.environ.get("RECEIPT2JSON_DATA"):
        cfg["data_dir"] = os.environ["RECEIPT2JSON_DATA"]

    return cfg


CFG = _load()


def data_dir() -> str:
    d = CFG["data_dir"]
    return d if os.path.isabs(d) else os.path.join(PROJECT_ROOT, d)


def max_upload_bytes() -> int:
    return int(CFG["max_upload_mb"]) * 1024 * 1024
