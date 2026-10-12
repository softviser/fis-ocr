"""Hedef JSON şeması — hem Ollama structured output'a verilir
hem de çıktı doğrulamasında kullanılır."""

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "qty": {"type": "number"},
        "unit_price": {"type": ["number", "null"]},
        "vat_rate": {"type": ["number", "null"]},
        "total": {"type": ["number", "null"]},
    },
    "required": ["name", "qty", "total"],
}

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor_name": {"type": ["string", "null"]},
        "vendor_tax_no": {"type": ["string", "null"]},
        "vendor_tax_office": {"type": ["string", "null"]},
        "vendor_address": {"type": ["string", "null"]},
        "vendor_phone": {"type": ["string", "null"]},
        "doc_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
        "doc_time": {"type": ["string", "null"], "description": "HH:MM"},
        "doc_no": {"type": ["string", "null"]},
        "register_no": {"type": ["string", "null"]},
        "category": {
            "type": "string",
            "enum": ["yazar_kasa", "e_arsiv", "e_fatura", "pos_slip", "diger"],
        },
        "payment_method": {
            "type": "string",
            "enum": ["kredi_karti", "nakit", "havale", "diger", "bilinmiyor"],
        },
        "currency": {"type": "string"},
        "items": {"type": "array", "items": ITEM_SCHEMA},
        "subtotal": {"type": ["number", "null"]},
        "vat_rate": {"type": ["number", "null"]},
        "vat_amount": {"type": ["number", "null"]},
        "grand_total": {"type": ["number", "null"]},
        "card_last4": {"type": ["string", "null"]},
        "card_bank": {"type": ["string", "null"]},
        "card_type": {"type": ["string", "null"]},
        "approval_code": {"type": ["string", "null"]},
        "batch_no": {"type": ["string", "null"]},
        "cashier": {"type": ["string", "null"]},
        "branch": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
    },
    # Ollama structured output yalnızca required alanların çıktıda yer almasını
    # garanti eder; tüm alanlar nullable olduğundan hepsini required yapıyoruz.
    "required": [
        "vendor_name", "vendor_tax_no", "vendor_tax_office", "vendor_address",
        "vendor_phone", "doc_date", "doc_time", "doc_no", "register_no",
        "category", "payment_method", "currency", "items", "subtotal",
        "vat_rate", "vat_amount", "grand_total", "card_last4", "card_bank",
        "card_type", "approval_code", "batch_no", "cashier", "branch", "notes",
    ],
}
