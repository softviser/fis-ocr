"""Komut satırı arayüzü.

Örnekler:
    python -m receipt2json.cli fisler/1.jpg
    python -m receipt2json.cli fisler/ --out cikti/
    python -m receipt2json.cli fisler/1.jpg --mode direct
    python -m receipt2json.cli fisler/1.jpg --ocr-model glm-ocr:latest --keep-ocr
"""
import argparse
import json
import os
import sys
import time

from .pipeline import ReceiptPipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def collect_images(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    return sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Türk fişi -> JSON (Ollama)")
    ap.add_argument("input", help="Görsel dosyası veya klasör")
    ap.add_argument("--out", help="Çıktı klasörü (yoksa stdout)")
    ap.add_argument("--host", default=None, help="Ollama adresi (örn. http://netviser:11434)")
    ap.add_argument("--mode", choices=["two_stage", "direct"], default="two_stage")
    ap.add_argument("--ocr-model", default=None, help="boşsa config.json'daki")
    ap.add_argument("--structure-model", default=None)
    ap.add_argument("--vision-model", default=None)
    ap.add_argument("--keep-ocr", action="store_true", help="_ocr_text alanını çıktıya ekle")
    args = ap.parse_args()

    images = collect_images(args.input)
    if not images:
        print(f"Görsel bulunamadı: {args.input}", file=sys.stderr)
        return 1

    pipe = ReceiptPipeline(
        host=args.host,
        ocr_model=args.ocr_model,
        structure_model=args.structure_model,
        vision_model=args.vision_model,
    )

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    failed = 0
    for img in images:
        t0 = time.time()
        try:
            docs = pipe.run_auto(img, mode=args.mode, keep_ocr=args.keep_ocr)
        except Exception as e:  # noqa: BLE001 - toplu işlemede devam et
            print(f"HATA  {img}: {e}", file=sys.stderr)
            failed += 1
            continue
        elapsed = time.time() - t0
        base = os.path.splitext(os.path.basename(img))[0]
        for i, doc in enumerate(docs, 1):
            out_json = json.dumps(doc, ensure_ascii=False, indent=4)
            if args.out:
                suffix = f"_{i}" if len(docs) > 1 else ""
                dest = os.path.join(args.out, f"{base}{suffix}.json")
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(out_json)
                print(f"OK    {img} -> {dest}  ({elapsed:.1f}s)")
            else:
                print(out_json)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
