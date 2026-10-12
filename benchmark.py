"""Hangi model senin fişlerinde daha iyi? Aynı görseli farklı yollardan geçirip
yan yana karşılaştır.

    python benchmark.py fis.jpg
    python benchmark.py fis.jpg --ocr-models deepseek-ocr:latest,glm-ocr:latest,minicpm-v4.5
    python benchmark.py fis.jpg --vision-models qwen3-vl:8b,llama3.2-vision:latest
    python benchmark.py fis.jpg --host http://10.0.0.100:11434
"""
import argparse
import json
import time

from receipt2json.pipeline import ReceiptPipeline

CRITICAL_FIELDS = [
    "vendor_name", "vendor_tax_no", "doc_date", "doc_time", "doc_no",
    "grand_total", "vat_amount", "card_last4", "approval_code", "payment_method",
]

DEFAULT_OCR = "deepseek-ocr:latest,glm-ocr:latest"
DEFAULT_VISION = "llama3.2-vision:latest"


def summarize(doc: dict) -> dict:
    return {k: doc.get(k) for k in CRITICAL_FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--host", default=None)
    ap.add_argument("--ocr-models", default=DEFAULT_OCR,
                    help="İki aşamalı yolda denenecek OCR modelleri (virgülle)")
    ap.add_argument("--vision-models", default=DEFAULT_VISION,
                    help="Tek aşamalı (direct) denenecek vision modelleri (virgülle)")
    ap.add_argument("--structure-model", default=None,
                    help="Yapılandırma modeli (boşsa config.json'daki)")
    args = ap.parse_args()

    pipe = ReceiptPipeline(host=args.host, structure_model=args.structure_model)

    runs = []
    for m in filter(None, (s.strip() for s in args.ocr_models.split(","))):
        runs.append((f"{m} + {pipe.structure_model}",
                     lambda m=m: pipe_run(pipe, args.image, m)))
    for m in filter(None, (s.strip() for s in args.vision_models.split(","))):
        runs.append((f"{m} (direkt)",
                     lambda m=m: direct_run(pipe, args.image, m)))

    for name, fn in runs:
        t0 = time.time()
        try:
            doc = fn()
            print(f"\n=== {name}  ({time.time()-t0:.1f}s) ===")
            print(json.dumps(summarize(doc), ensure_ascii=False, indent=2))
            if doc.get("_suspect"):
                print("!! SÜPHELİ (topraklama ihlali)")
            if doc.get("_warnings"):
                print("UYARILAR:", doc["_warnings"])
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {name} === HATA: {e}")


def pipe_run(pipe: ReceiptPipeline, image: str, ocr_model: str) -> dict:
    text = pipe.ocr(image, model=ocr_model)
    doc = pipe.structure(text)
    return pipe.postprocess(doc, text)


def direct_run(pipe: ReceiptPipeline, image: str, model: str) -> dict:
    old = pipe.vision_model
    pipe.vision_model = model
    try:
        doc = pipe.extract_direct(image)
        return pipe.postprocess(doc)
    finally:
        pipe.vision_model = old


if __name__ == "__main__":
    main()
