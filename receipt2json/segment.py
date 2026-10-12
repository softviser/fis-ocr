"""Çok fişli görseli tek tek fişlere bölme (saf Pillow + Python).

Yöntem:
  1. Yerel (adaptif) eşikleme ile YAZI maskesi çıkarılır — global eşiğin
     aksine karanlık zemin, gölge ve ışık farklarından etkilenmez.
  2. Sütun projeksiyonu: fişler arasındaki boşlukta yazı olmaz (kağıtlar
     bitişik olsa bile kenar marjları boşluk bırakır); yazısız dikey şeritler
     fiş sütunlarını ayırır.
  3. Projeksiyon bölemediyse `looks_multi` sahneyi sınıflandırır: sütun
     projeksiyonunda 2+ belirgin vadi "masaya serilmiş çok fiş" demektir.
     Bu durumda pipeline, OCR'a girmeden VLM kutu tespitine gider
     (şaşırtmalı/eğik ızgara dizilimlerini projeksiyon ayıramaz: fişler
     arası boşluk görüntü boyunca hizalı değildir, buruşuk kağıtlardaki
     kat izi gölgeleri de sahte "boşluk" üretir).
  4. İkinci tur (pipeline tarafından çağrılır): bir kırpmanın OCR'ında 2+
     TOPLAM görülürse `split_rows` / `split_cols` ile tekrar bölünür.
     Fiş + banka slibi tek TOPLAM içerdiğinden asla yanlış bölünmez.

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

MARGIN = 12  # kırpma payı (orijinal piksel)


# ---------------------------------------------------------------- ortak

def _load_small(image_path: str, work_width: int = 300):
    im = Image.open(image_path).convert("L")
    w, h = im.size
    scale = max(1.0, w / work_width)
    sw, sh = max(1, int(w / scale)), max(1, int(h / scale))
    px = list(im.resize((sw, sh)).getdata())
    return px, sw, sh, scale, (w, h)


def _local_text_mask(px, sw, sh, window_ratio: float = 0.08,
                     factor: float = 0.82) -> list[int]:
    """Yerel eşikleme: piksel, çevresinin ortalamasından belirgin koyuysa
    yazıdır. Integral görüntü ile O(n)."""
    r = max(3, int(min(sw, sh) * window_ratio))
    integ = [0] * ((sw + 1) * (sh + 1))
    for y in range(sh):
        row_sum = 0
        base = y * sw
        ibase = (y + 1) * (sw + 1)
        pbase = y * (sw + 1)
        for x in range(sw):
            row_sum += px[base + x]
            integ[ibase + x + 1] = integ[pbase + x + 1] + row_sum
    mask = [0] * (sw * sh)
    for y in range(sh):
        y0, y1 = max(0, y - r), min(sh - 1, y + r)
        for x in range(sw):
            x0, x1 = max(0, x - r), min(sw - 1, x + r)
            n = (x1 - x0 + 1) * (y1 - y0 + 1)
            s = (integ[(y1 + 1) * (sw + 1) + x1 + 1]
                 - integ[y0 * (sw + 1) + x1 + 1]
                 - integ[(y1 + 1) * (sw + 1) + x0]
                 + integ[y0 * (sw + 1) + x0])
            v = px[y * sw + x]
            if v < (s / n) * factor and v < 210:
                mask[y * sw + x] = 1
    return mask


def _bands(counts: list[int], eps: int, min_gap: int, min_band: int):
    """Projeksiyonda içerik barındıran aralıklar; eps altı uzun boşluklar böler.

    Tek pikselllik pürüzler yumuşatılır; min_band'den dar bantlar komşusuna
    birleştirilir (içerik atılmaz — fişin kenar sütunu ayrı fiş sanılmasın).
    """
    n = len(counts)
    sm = [
        (counts[max(0, i - 1)] + counts[i] + counts[min(n - 1, i + 1)]) // 3
        for i in range(n)
    ]

    bands = []
    start = None
    gap = 0
    for i, c in enumerate(sm):
        if c > eps:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                bands.append((start, i - gap))
                start, gap = None, 0
    if start is not None:
        end = n - 1
        while end > start and sm[end] <= eps:
            end -= 1
        bands.append((start, end))

    # Dar bantları en yakın komşuya birleştir (aradaki boşluk köprülenir)
    changed = True
    while changed and len(bands) > 1:
        changed = False
        for i, (b0, b1) in enumerate(bands):
            if b1 - b0 + 1 >= min_band:
                continue
            left_gap = b0 - bands[i - 1][1] if i > 0 else None
            right_gap = bands[i + 1][0] - b1 if i < len(bands) - 1 else None
            if left_gap is not None and (right_gap is None or left_gap <= right_gap):
                bands[i - 1] = (bands[i - 1][0], b1)
            else:
                bands[i + 1] = (b0, bands[i + 1][1])
            del bands[i]
            changed = True
            break
    return [b for b in bands if b[1] - b[0] + 1 >= min_band]


def _tight_y(mask, sw, sh, x0, x1):
    ys = [y for y in range(sh)
          if any(mask[y * sw + x] for x in range(x0, x1 + 1))]
    return (min(ys), max(ys)) if ys else (0, sh - 1)


def _crop_boxes(image_path, boxes_small, scale, size, out_dir, suffix):
    w, h = size
    out_dir = out_dir or tempfile.mkdtemp(prefix="r2j_seg_")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    im = Image.open(image_path)
    paths = []
    for i, (x0, y0, x1, y1) in enumerate(boxes_small, 1):
        box = (max(0, int(x0 * scale) - MARGIN),
               max(0, int(y0 * scale) - MARGIN),
               min(w, int((x1 + 1) * scale) + MARGIN),
               min(h, int((y1 + 1) * scale) + MARGIN))
        dest = os.path.join(out_dir, f"{base}_{suffix}{i}.png")
        im.crop(box).save(dest)
        paths.append(dest)
    return paths


# ---------------------------------------------------------------- sahne sınıflandırma

def looks_multi(image_path: str) -> bool:
    """Fotoğraf çok fişli bir 'masa' sahnesi gibi mi görünüyor?

    Sütun projeksiyonunda tepe -> derin çukur -> tepe deseni (vadi) komşu
    fiş sütunlarının izidir. Tek fiş yakın çekimi 0-1 vadi üretir (etiket
    ile değer sütunu arasındaki boşluk); 2+ vadi birden fazla fiş demektir.
    Projeksiyon vadileri ayırmaya yetmese bile (eğik/şaşırtmalı dizilim)
    saymaya yeter — bu yüzden `split_receipts` bölemese de bu sezgi tutar.
    """
    if not HAS_PIL:
        return False
    px, sw, sh, _, _ = _load_small(image_path)
    mask = _local_text_mask(px, sw, sh)
    counts = [sum(mask[y * sw + x] for y in range(sh)) for x in range(sw)]
    k = 3
    sm = [
        sum(counts[max(0, i - k):i + k + 1]) / (min(sw - 1, i + k) - max(0, i - k) + 1)
        for i in range(sw)
    ]
    peak = sorted(sm)[int(sw * 0.9)]
    if peak <= 0:
        return False
    lo_thr, hi_thr = peak * 0.35, peak * 0.70
    valleys = 0
    state = "start"
    for v in sm:
        if state != "high" and v >= hi_thr:
            if state == "low":
                valleys += 1
            state = "high"
        elif state == "high" and v <= lo_thr:
            state = "low"
    return valleys >= 2


def crop_norm_boxes(image_path: str, boxes: list[dict],
                    out_dir: str | None = None) -> list[str]:
    """0-1000 normalize sınırlayıcı kutuları doğrulayıp kırp (VLM tespiti için).

    - ters koordinatlar düzeltilir; 1000'i aşan değerler piksel koordinatı
      sayılıp görüntü boyutuna göre normalize edilir
    - çok küçük kutular ve metin içermeyenler (boş masa/gölge) elenir
    - %60+ örtüşen mükerrer kutular teke indirilir
    Dönen liste boş olabilir; çağıran tarafın yedeğe düşmesi beklenir.
    """
    if not HAS_PIL or not boxes:
        return []
    px, sw, sh, scale, size = _load_small(image_path)
    mask = _local_text_mask(px, sw, sh)
    w, h = size

    norm = []
    raw_max = max((max(b["x0"], b["y0"], b["x1"], b["y1"]) for b in boxes),
                  default=0)
    for b in boxes:
        x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
        if raw_max > 1000:  # model piksel koordinatı dönmüş
            x0, x1 = x0 * 1000 // max(1, w), x1 * 1000 // max(1, w)
            y0, y1 = y0 * 1000 // max(1, h), y1 * 1000 // max(1, h)
        x0, x1 = sorted((max(0, min(1000, x0)), max(0, min(1000, x1))))
        y0, y1 = sorted((max(0, min(1000, y0)), max(0, min(1000, y1))))
        if x1 - x0 < 50 or y1 - y0 < 50:  # %5'ten küçük kutu anlamsız
            continue
        norm.append((x0, y0, x1, y1))

    # metin doğrulaması: kutu içinde yeterli maske pikseli olmalı
    def text_px(x0, y0, x1, y1):
        cx0, cx1 = x0 * sw // 1000, max(x0 * sw // 1000 + 1, x1 * sw // 1000)
        cy0, cy1 = y0 * sh // 1000, max(y0 * sh // 1000 + 1, y1 * sh // 1000)
        n = sum(mask[y * sw + x]
                for y in range(cy0, min(sh, cy1))
                for x in range(cx0, min(sw, cx1)))
        return n, max(1, (cx1 - cx0) * (cy1 - cy0))

    valid = []
    for box in norm:
        n, area = text_px(*box)
        if n / area >= 0.015:
            valid.append(box)

    # mükerrer eleme (IoU > 0.6); büyük kutu kalır
    valid.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: list[tuple] = []
    for b in valid:
        dup = False
        for o in kept:
            ix = max(0, min(b[2], o[2]) - max(b[0], o[0]))
            iy = max(0, min(b[3], o[3]) - max(b[1], o[1]))
            inter = ix * iy
            union = ((b[2] - b[0]) * (b[3] - b[1])
                     + (o[2] - o[0]) * (o[3] - o[1]) - inter)
            if union and inter / union > 0.6:
                dup = True
                break
        if not dup:
            kept.append(b)
    kept = kept[:16]  # makul üst sınır
    kept.sort(key=lambda b: (b[1] // 100, b[0]))  # okuma sırası

    margin = 15  # normalize birim
    im = Image.open(image_path)
    out_dir = out_dir or tempfile.mkdtemp(prefix="r2j_vlm_")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    paths = []
    for i, (x0, y0, x1, y1) in enumerate(kept, 1):
        px0 = max(0, (x0 - margin) * w // 1000)
        py0 = max(0, (y0 - margin) * h // 1000)
        px1 = min(w, (x1 + margin) * w // 1000)
        py1 = min(h, (y1 + margin) * h // 1000)
        dest = os.path.join(out_dir, f"{base}_vlm{i}.png")
        im.crop((px0, py0, px1, py1)).save(dest)
        paths.append(dest)
    return paths


# ---------------------------------------------------------------- birincil

def split_receipts(image_path: str, out_dir: str | None = None) -> list[str]:
    """Görseli fiş sütunlarına böl (yan yana fişler).

    0 veya 1 sütun bulunursa orijinal yol tek elemanlı liste olarak döner.
    Alt alta duran ayrı fişler burada DEĞİL, pipeline'ın ikinci turunda
    (TOPLAM sayısı kanıtıyla, `split_rows` ile) ayrılır.
    """
    if not HAS_PIL:
        return [image_path]
    px, sw, sh, scale, size = _load_small(image_path)
    mask = _local_text_mask(px, sw, sh)

    col_counts = [sum(mask[y * sw + x] for y in range(sh)) for x in range(sw)]
    bands = _bands(
        col_counts,
        eps=max(1, int(sh * 0.012)),
        min_gap=max(2, int(sw * 0.012)),
        min_band=int(sw * 0.15),
    )
    if len(bands) < 2:
        return [image_path]

    boxes = []
    for (x0, x1) in bands:
        y0, y1 = _tight_y(mask, sw, sh, x0, x1)
        boxes.append((x0, y0, x1, y1))
    paths = _crop_boxes(image_path, boxes, scale, size, out_dir, "fis")
    log.info("%s -> %d fiş sütunu", image_path, len(paths))
    return paths


# ---------------------------------------------------------------- ikinci tur

def split_rows(image_path: str, out_dir: str | None = None,
               max_parts: int = 4) -> list[str]:
    """Kırpmayı yatay boşluklardan dikeyde böl (alt alta duran ayrı fişler)."""
    return _split_axis(image_path, axis="y", out_dir=out_dir, max_parts=max_parts)


def split_cols(image_path: str, out_dir: str | None = None,
               max_parts: int = 4) -> list[str]:
    """Kırpmayı daha toleranslı eşikle dikey şeritlere böl (yan yana fişler)."""
    return _split_axis(image_path, axis="x", out_dir=out_dir, max_parts=max_parts)


def _split_axis(image_path, axis, out_dir, max_parts):
    if not HAS_PIL:
        return [image_path]
    px, sw, sh, scale, size = _load_small(image_path)
    mask = _local_text_mask(px, sw, sh)

    if axis == "y":
        counts = [sum(mask[y * sw + x] for x in range(sw)) for y in range(sh)]
        span, other = sh, sw
    else:
        counts = [sum(mask[y * sw + x] for y in range(sh)) for x in range(sw)]
        span, other = sw, sh

    bands = _bands(
        counts,
        eps=max(1, int(other * 0.015)),
        min_gap=max(2, int(span * 0.01)),
        min_band=int(span * 0.08),
    )
    if len(bands) < 2:
        return [image_path]
    bands = bands[:max_parts]

    boxes = []
    for (b0, b1) in bands:
        if axis == "y":
            xs = [x for x in range(sw) if any(
                mask[y * sw + x] for y in range(b0, b1 + 1))]
            x0, x1 = (min(xs), max(xs)) if xs else (0, sw - 1)
            boxes.append((x0, b0, x1, b1))
        else:
            y0, y1 = _tight_y(mask, sw, sh, b0, b1)
            boxes.append((b0, y0, b1, y1))
    suffix = "satir" if axis == "y" else "sutun"
    return _crop_boxes(image_path, boxes, scale, size, out_dir, suffix)
