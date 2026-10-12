"""Bölütleme birim testleri: looks_multi + crop_norm_boxes + vlm_split.

Ollama gerektirmez; sentetik görseller ve sahte VLM yanıtı kullanır.

    py -3 tests/test_segment.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw  # noqa: E402

from receipt2json import segment  # noqa: E402


def _draw_receipt(d, x0, y0, x1, y1):
    """Beyaz kağıt + içine metin benzeri koyu satırlar çiz."""
    d.rectangle([x0, y0, x1, y1], fill=(250, 250, 248))
    y = y0 + 20
    while y < y1 - 20:
        d.rectangle([x0 + 12, y, x1 - 12, y + 6], fill=(40, 40, 40))
        y += 18


def make_single(path):
    im = Image.new("RGB", (600, 1400), (170, 170, 168))  # masa
    d = ImageDraw.Draw(im)
    _draw_receipt(d, 60, 60, 540, 1340)
    im.save(path)


def make_table(path):
    """3 sütun x 2 sıra fiş serili masa."""
    im = Image.new("RGB", (1500, 1200), (170, 170, 168))
    d = ImageDraw.Draw(im)
    for i, x in enumerate((60, 560, 1060)):
        _draw_receipt(d, x, 60, x + 380, 560)
        _draw_receipt(d, x + 10, 640, x + 390, 1140)
    im.save(path)
    return im.size


def main():
    tmp = tempfile.mkdtemp(prefix="r2j_segtest_")
    single = os.path.join(tmp, "tek.png")
    table = os.path.join(tmp, "masa.png")
    make_single(single)
    w, h = make_table(table)

    # --- looks_multi
    assert not segment.looks_multi(single), "tek fiş çok fişli sanıldı"
    assert segment.looks_multi(table), "masa sahnesi tek fiş sanıldı"
    print("looks_multi tamam")

    # --- crop_norm_boxes: normalize kutular
    boxes = [
        {"x0": 40, "y0": 50, "x1": 293, "y1": 467},     # sol üst fiş
        {"x0": 373, "y0": 50, "x1": 627, "y1": 467},    # orta üst fiş
        {"x0": 380, "y0": 55, "x1": 630, "y1": 470},    # mükerrer (IoU yüksek)
        {"x0": 300, "y0": 480, "x1": 360, "y1": 530},   # boş masa parçası
        {"x0": 10, "y0": 10, "x1": 40, "y1": 35},       # çok küçük
        {"x0": 973, "y0": 467, "x1": 707, "y1": 533},   # ters koordinat + boş alan
    ]
    paths = segment.crop_norm_boxes(table, boxes, out_dir=os.path.join(tmp, "k1"))
    assert len(paths) == 2, f"2 geçerli kutu beklenirdi, {len(paths)} çıktı"
    for p in paths:
        assert os.path.exists(p)
        cw, ch = Image.open(p).size
        assert cw > 100 and ch > 100
    print("crop_norm_boxes (normalize) tamam")

    # --- crop_norm_boxes: piksel koordinatı dönen model
    pix_boxes = [
        {"x0": 60, "y0": 60, "x1": 440, "y1": 560},
        {"x0": 560, "y0": 60, "x1": 940, "y1": 560},
        {"x0": 1060, "y0": 60, "x1": 1440, "y1": 560},
    ]
    paths = segment.crop_norm_boxes(table, pix_boxes, out_dir=os.path.join(tmp, "k2"))
    assert len(paths) == 3, f"3 piksel kutusu beklenirdi, {len(paths)} çıktı"
    print("crop_norm_boxes (piksel) tamam")

    # --- vlm_split: sahte VLM yanıtı ile uçtan uca
    from receipt2json.pipeline import ReceiptPipeline

    p = ReceiptPipeline()
    p.client.chat = lambda *a, **k: json.dumps({"receipts": boxes})
    parts = p.vlm_split(table)
    assert len(parts) == 2, f"vlm_split 2 kırpma dönmeliydi: {parts}"

    def boom(*a, **k):
        raise RuntimeError("ollama yok")

    p.client.chat = boom
    assert p.vlm_split(table) == [table], "VLM hatasında orijinal dönmeli"
    print("vlm_split tamam")

    print("TUM BOLUTLEME TESTLERI GECTI")


if __name__ == "__main__":
    main()
