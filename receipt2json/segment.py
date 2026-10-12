"""Çok fişli görseli tek tek fişlere bölme (saf Pillow + Python).

Yöntem: Otsu eşikleme -> polarite seçimi (arka plan, görüntü kenarına dokunan
çoğunluk sınıfıdır; fişler diğer sınıf) -> fiş maskesinde bağlı bileşen
analizi -> bulunamazsa asimetrik genleşme (dikeyde güçlü/yatayda zayıf) ile
yazı bloklarını fiş sütunlarına birleştirme -> alt alta blokları tek fişte
toplama (fiskal kısım + banka slibi aynı kağıttadır).

Pillow kurulu değilse görsel tek parça döner; pipeline bozulmaz (çok fişli
kareler groundedness kontrolünde `_suspect` olarak işaretlenir).
"""
import logging
import os
import tempfile

log = logging.getLogger("receipt2json.segment")

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

MIN_AREA_RATIO = 0.04   # fiş adayı en az görüntünün %4'ü olmalı
MIN_ASPECT = 0.8        # yükseklik/genişlik (fişler dikine uzun ya da kareye yakın)
MARGIN = 12             # kırpma payı (orijinal piksel)


def split_receipts(image_path: str, out_dir: str | None = None) -> list[str]:
    """Görseldeki fiş bölgelerini bulup ayrı dosyalara kırpar.

    0 veya 1 bölge bulunursa orijinal yol tek elemanlı liste olarak döner.
    """
    if not HAS_PIL:
        return [image_path]

    boxes, size = _boxes_pil(image_path)
    if len(boxes) <= 1:
        return [image_path]

    boxes.sort(key=lambda b: b[0])  # soldan sağa
    out_dir = out_dir or tempfile.mkdtemp(prefix="r2j_seg_")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    w, h = size

    paths = []
    im = Image.open(image_path)
    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        box = (max(0, x0 - MARGIN), max(0, y0 - MARGIN),
               min(w, x1 + MARGIN), min(h, y1 + MARGIN))
        dest = os.path.join(out_dir, f"{base}_fis{i}.png")
        im.crop(box).save(dest)
        paths.append(dest)

    log.info("%s -> %d fiş bölgesi", image_path, len(paths))
    return paths


def _boxes_pil(image_path: str, work_width: int = 300):
    im = Image.open(image_path).convert("L")
    w, h = im.size
    scale = max(1.0, w / work_width)
    sw, sh = max(1, int(w / scale)), max(1, int(h / scale))
    small = im.resize((sw, sh))
    px = list(small.getdata())

    thr = _otsu(px)
    dark = [1 if v < thr else 0 for v in px]

    # Polarite: görüntü kenarındaki piksellerin çoğunluğu arka plandır
    border = (
        [dark[x] for x in range(sw)]
        + [dark[(sh - 1) * sw + x] for x in range(sw)]
        + [dark[y * sw] for y in range(sh)]
        + [dark[y * sw + sw - 1] for y in range(sh)]
    )
    bg_is_dark = sum(border) * 2 > len(border)
    mask = [1 - v for v in dark] if bg_is_dark else dark

    # Faz A: fiş gövdeleri (kağıt arka plandan ayrışıyorsa) doğrudan bulunur
    boxes = _filter_components(_components(mask, sw, sh), sw, sh, min_fill=0.45)

    if len(boxes) < 2:
        # Faz B: maske yazı piksellerinden ibaretse (kağıt da arka plan kadar
        # parlaksa) fiş gövdeleri bulunamaz. Dikeyde güçlü / yatayda zayıf
        # genleşme ile her fişin yazıları tek bloba birleştirilir; fişler
        # arasındaki yatay boşluk korunur. Boşluk genişliği görüntüden
        # görüntüye değiştiği için birkaç yarıçap denenir, en çok fiş bulan
        # sonuç alınır (tek fişli görseller her yarıçapta tek blok kalır).
        best: list = []
        for rv, rh in ((4, 1), (6, 2), (8, 1)):
            dilated = _dilate(mask, sw, sh, rv=rv, rh=rh)
            boxes_b = _filter_components(
                _components(dilated, sw, sh), sw, sh, min_fill=0.30
            )
            if len(boxes_b) > len(best):
                best = boxes_b
        if len(best) >= 2:
            # Aynı sütunda ALT ALTA duran bloklar tek fiştir (fiskal kısım +
            # banka slibi); yan yana veya iç içe olanlar ayrı kalır
            best = _merge_stacked(best)
        if len(best) >= 2:
            # Her kutuyu kendi x-aralığındaki TÜM yazı satırlarını kapsayacak
            # şekilde dikeyde genişlet (POS slibi gibi alt bölümler genleşmede
            # komşu bloba karışıp kutu dışında kalabiliyor)
            boxes = [_expand_y(mask, sw, sh, b) for b in best]

    return [
        (int(x0 * scale), int(y0 * scale),
         int((x1 + 1) * scale), int((y1 + 1) * scale))
        for (x0, y0, x1, y1) in boxes
    ], (w, h)


def _filter_components(comps, sw, sh, min_fill: float):
    boxes = []
    for (x0, y0, x1, y1, area) in comps:
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if area / (sw * sh) < MIN_AREA_RATIO:
            continue
        if bh / max(bw, 1) < MIN_ASPECT:
            continue
        if area / (bw * bh) < min_fill:
            continue
        boxes.append((x0, y0, x1, y1))
    return boxes


def _merge_stacked(boxes):
    """Alt alta duran kutuları birleştir: x-aralıkları büyük oranda örtüşen
    AMA y-aralıkları ayrık olan kutular aynı fişin bölümleridir.

    (y'leri de örtüşenler iç içe/karışık kutulardır; onlara dokunulmaz.)
    """
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax0, ay0, ax1, ay1 = boxes[i]
                bx0, by0, bx1, by1 = boxes[j]
                x_overlap = min(ax1, bx1) - max(ax0, bx0)
                min_w = min(ax1 - ax0, bx1 - bx0)
                y_overlap = min(ay1, by1) - max(ay0, by0)
                min_h = min(ay1 - ay0, by1 - by0)
                if (min_w > 0 and x_overlap / min_w > 0.5
                        and (min_h <= 0 or y_overlap / min_h < 0.5)):
                    boxes[i] = (min(ax0, bx0), min(ay0, by0),
                                max(ax1, bx1), max(ay1, by1))
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def _expand_y(mask: list[int], sw: int, sh: int, box):
    """Kutunun y sınırlarını, x-aralığında maske içeren tüm satırlara genişlet."""
    x0, y0, x1, y1 = box
    ny0, ny1 = y0, y1
    for y in range(sh):
        base = y * sw
        if any(mask[base + x] for x in range(x0, x1 + 1)):
            ny0 = min(ny0, y)
            ny1 = max(ny1, y)
    return (x0, ny0, x1, ny1)


def _otsu(pixels: list[int]) -> int:
    hist = [0] * 256
    for v in pixels:
        hist[v] += 1
    total = len(pixels)
    sum_all = sum(i * hist[i] for i in range(256))
    sum_b = 0.0
    w_b = 0
    best_thr, best_var = 127, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_thr = var, t
    return best_thr


def _components(mask: list[int], w: int, h: int):
    """4-komşuluk, yinelemeli (stack) bağlı bileşen: (x0,y0,x1,y1,alan) listesi."""
    seen = bytearray(w * h)
    out = []
    for start in range(w * h):
        if seen[start] or not mask[start]:
            continue
        stack = [start]
        seen[start] = 1
        x0 = x1 = start % w
        y0 = y1 = start // w
        area = 0
        while stack:
            idx = stack.pop()
            area += 1
            x, y = idx % w, idx // w
            if x < x0: x0 = x
            if x > x1: x1 = x
            if y < y0: y0 = y
            if y > y1: y1 = y
            if x > 0 and mask[idx - 1] and not seen[idx - 1]:
                seen[idx - 1] = 1; stack.append(idx - 1)
            if x < w - 1 and mask[idx + 1] and not seen[idx + 1]:
                seen[idx + 1] = 1; stack.append(idx + 1)
            if y > 0 and mask[idx - w] and not seen[idx - w]:
                seen[idx - w] = 1; stack.append(idx - w)
            if y < h - 1 and mask[idx + w] and not seen[idx + w]:
                seen[idx + w] = 1; stack.append(idx + w)
        out.append((x0, y0, x1, y1, area))
    return out


def _dilate(mask: list[int], w: int, h: int, rv: int, rh: int) -> list[int]:
    """Ayrıştırılabilir genleşme: önce dikey ±rv, sonra yatay ±rh."""
    tmp = [0] * (w * h)
    for x in range(w):
        run = 0
        for y in range(h):
            if mask[y * w + x]:
                run = 2 * rv + 1
            elif run:
                run -= 1
            if run > rv:
                tmp[y * w + x] = 1
        run = 0
        for y in range(h - 1, -1, -1):
            if mask[y * w + x]:
                run = rv + 1
            elif run:
                run -= 1
            if run:
                tmp[y * w + x] = 1
    out = [0] * (w * h)
    for y in range(h):
        base = y * w
        run = 0
        for x in range(w):
            if tmp[base + x]:
                run = 2 * rh + 1
            elif run:
                run -= 1
            if run > rh:
                out[base + x] = 1
        run = 0
        for x in range(w - 1, -1, -1):
            if tmp[base + x]:
                run = rh + 1
            elif run:
                run -= 1
            if run:
                out[base + x] = 1
    return out
