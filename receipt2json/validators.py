"""Deterministik doğrulama katmanı: VKN/TCKN sağlama, aritmetik kontrol,
kayıtlı firma/kart eşleştirme."""
import json
import os
import re
from datetime import datetime


# ---------------------------------------------------------------- VKN / TCKN

def vkn_valid(vkn: str) -> bool:
    """10 haneli Vergi Kimlik Numarası sağlama algoritması."""
    if not vkn or not re.fullmatch(r"\d{10}", vkn):
        return False
    digits = [int(c) for c in vkn]
    total = 0
    for i in range(9):
        p = (digits[i] + 9 - i) % 10
        q = (p * (2 ** (9 - i))) % 9
        if p != 0 and q == 0:
            q = 9
        total += q
    check = (10 - total % 10) % 10
    return check == digits[9]


def tckn_valid(tckn: str) -> bool:
    """11 haneli TC Kimlik Numarası sağlama algoritması (şahıs firmaları için)."""
    if not tckn or not re.fullmatch(r"\d{11}", tckn) or tckn[0] == "0":
        return False
    d = [int(c) for c in tckn]
    d10 = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    d11 = sum(d[:10]) % 10
    return d[9] == d10 and d[10] == d11


def tax_no_valid(tax_no: str) -> bool:
    if not tax_no:
        return False
    tax_no = re.sub(r"\D", "", tax_no)
    if len(tax_no) == 10:
        return vkn_valid(tax_no)
    if len(tax_no) == 11:
        return tckn_valid(tax_no)
    return False


# ---------------------------------------------------------------- aritmetik

def arithmetic_check(doc: dict, tolerance: float = 0.05) -> list[str]:
    """Toplam/KDV/kalem tutarlılığını kontrol et; sorunların listesini döndür."""
    problems: list[str] = []
    grand = doc.get("grand_total")
    vat = doc.get("vat_amount")
    sub = doc.get("subtotal")
    rate = doc.get("vat_rate")
    items = doc.get("items") or []

    if grand is not None and vat is not None:
        if sub is None:
            doc["subtotal"] = round(grand - vat, 2)
            sub = doc["subtotal"]
        elif abs((sub + vat) - grand) > tolerance:
            problems.append(
                f"subtotal({sub}) + vat_amount({vat}) != grand_total({grand})"
            )

    # KDV dahil fiyatta beklenen KDV: grand * rate / (100 + rate)
    if grand is not None and vat is not None and rate:
        expected_vat = round(grand * rate / (100 + rate), 2)
        if abs(expected_vat - vat) > max(tolerance, grand * 0.002):
            problems.append(
                f"vat_amount({vat}) beklenen değerle uyuşmuyor (~{expected_vat})"
            )

    if items and grand is not None:
        items_total = sum(i.get("total") or 0 for i in items)
        if items_total and abs(items_total - grand) > tolerance:
            problems.append(
                f"kalem toplamı({round(items_total, 2)}) != grand_total({grand})"
            )
    return problems


# ---------------------------------------------------------------- topraklama

def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _amount_in_text(amount: float, ocr_digits: str) -> bool:
    """590.0 -> '59000' (kuruşlu) veya '590' OCR rakam akışında geçiyor mu?"""
    if amount is None:
        return True
    cents = str(int(round(amount * 100)))
    plain = str(int(amount)) if float(amount).is_integer() else None
    return cents in ocr_digits or (plain is not None and plain in ocr_digits)


def groundedness_check(doc: dict, ocr_text: str) -> tuple[list[str], bool]:
    """Modelin yazdığı kritik değerler OCR metninde gerçekten var mı?

    Dönüş: (uyarılar, süpheli_mi). LLM halüsinasyonuna karşı deterministik fren.
    """
    warnings: list[str] = []
    suspect = False

    if not ocr_text or len(ocr_text.strip()) < 40:
        return (["OCR metni çok kısa/boş; çıktı güvenilmez"], True)

    ocr_digits = _digits_only(ocr_text)
    upper = ocr_text.upper()

    # Kritik rakamsal alanlar metinde geçmeli
    for field in ("vendor_tax_no", "doc_no", "approval_code", "card_last4"):
        val = doc.get(field)
        if val and _digits_only(str(val)) and _digits_only(str(val)) not in ocr_digits:
            warnings.append(f"{field}='{val}' OCR metninde yok")
            suspect = True

    for field in ("grand_total", "vat_amount"):
        val = doc.get(field)
        if val is not None and not _amount_in_text(val, ocr_digits):
            warnings.append(f"{field}={val} OCR metninde yok")
            suspect = True

    # Firma adının en az bir kelimesi metinde geçmeli
    name = (doc.get("vendor_name") or "").upper()
    words = [w for w in re.split(r"\W+", name) if len(w) >= 4]
    if words and not any(w in upper for w in words):
        warnings.append(f"vendor_name='{doc.get('vendor_name')}' OCR metninde yok")
        suspect = True

    # Aynı karede birden fazla fiş işareti
    toplam_count = len(re.findall(r"TOPLAM", upper))
    tarih_count = len(re.findall(r"TAR[İI]H", upper))
    if toplam_count >= 2 or tarih_count >= 2:
        warnings.append(
            f"Görselde birden fazla fiş olabilir (TOPLAM x{toplam_count}, "
            f"TARİH x{tarih_count}); alanlar karışmış olabilir — görseli tek fiş "
            "olacak şekilde bölüp tekrar deneyin"
        )
        suspect = True

    return warnings, suspect


# ---------------------------------------------------------------- normalize

_DATE_PATTERNS = ["%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%Y-%m-%d"]


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    for pat in _DATE_PATTERNS:
        try:
            return datetime.strptime(value, pat).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value  # çevrilemedi, olduğu gibi bırak


def normalize(doc: dict) -> dict:
    doc["doc_date"] = normalize_date(doc.get("doc_date"))
    if doc.get("doc_time"):
        m = re.match(r"(\d{1,2}):(\d{2})", str(doc["doc_time"]))
        doc["doc_time"] = f"{int(m.group(1)):02d}:{m.group(2)}" if m else doc["doc_time"]
    if doc.get("vendor_tax_no"):
        doc["vendor_tax_no"] = re.sub(r"\D", "", str(doc["vendor_tax_no"]))
    if doc.get("card_last4"):
        digits = re.sub(r"\D", "", str(doc["card_last4"]))
        doc["card_last4"] = digits[-4:] if digits else None
    if not doc.get("currency"):
        doc["currency"] = "TRY"
    return doc


# ---------------------------------------------------------------- lookup DB

def _load_db(path: str) -> dict:
    """Veritabanını yükle; yanında .local.json varsa üstüne birleştir.

    Örn. vendors.json (repoya giren örnekler) + vendors.local.json
    (gitignore'da, gerçek kayıtlar).
    """
    db: dict = {}
    local = os.path.splitext(path)[0] + ".local.json"
    for p in (path, local):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                db.update(json.load(f))
    return db


def tax_check(doc: dict, vendors_db_path: str) -> dict:
    """_tax_check alanını üret: sağlama + kayıtlı firma eşleştirme."""
    original = doc.get("vendor_tax_no")
    valid = tax_no_valid(original) if original else False
    vendors = _load_db(vendors_db_path)  # {"vkn": {"title": ..., "source": ...}}

    result = {
        "status": "valid" if valid else ("invalid" if original else "missing"),
        "original": original,
        "valid": valid,
        "effective": original if valid else None,
        "suggestion": None,
        "suggestion_title": None,
        "source": None,
        "reason": "Sağlama geçerli" if valid else "Sağlama başarısız",
    }

    if original and original in vendors:
        entry = vendors[original]
        result["suggestion_title"] = entry.get("title")
        result["source"] = entry.get("source")
        result["reason"] = f"Kayıtlı: {entry.get('title')}"
    elif not valid and original and vendors:
        # OCR hatası olabilir: tek karakter farklı kayıtlı VKN ara
        for vkn, entry in vendors.items():
            if len(vkn) == len(original) and sum(
                a != b for a, b in zip(vkn, original)
            ) == 1:
                result["suggestion"] = vkn
                result["suggestion_title"] = entry.get("title")
                result["source"] = entry.get("source")
                result["effective"] = vkn
                result["reason"] = f"OCR hatası olasılığı; kayıtlı benzer VKN: {vkn}"
                break
    return result


def card_hint(doc: dict, cards_db_path: str) -> str | None:
    """_card_hint alanını üret: son 4 haneden kayıtlı kart eşleştirme."""
    last4 = doc.get("card_last4")
    if not last4:
        return None
    cards = _load_db(cards_db_path)  # {"5023": "GARANTİ SS - xxxx xxxx 5023"}
    return cards.get(last4)
