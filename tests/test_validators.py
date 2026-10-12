"""Deterministik katman testleri — model gerektirmez.

    python -m pytest tests/ -q   (veya)   python tests/test_validators.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from receipt2json import validators  # noqa: E402


def test_vkn_checksum():
    assert validators.vkn_valid("3620273655")
    assert validators.vkn_valid("5820492073")
    assert not validators.vkn_valid("3620273656")
    assert not validators.vkn_valid("123")


def test_date_normalize():
    assert validators.normalize_date("20/08/2026") == "2026-08-20"
    assert validators.normalize_date("05.08.2026") == "2026-08-05"
    assert validators.normalize_date("2026-08-20") == "2026-08-20"


def test_arithmetic():
    doc = {"grand_total": 590.0, "vat_amount": 53.64, "subtotal": None,
           "vat_rate": 10, "items": [{"total": 590.0}]}
    problems = validators.arithmetic_check(doc)
    assert doc["subtotal"] == 536.36
    assert problems == []

    bad = {"grand_total": 590.0, "vat_amount": 99.0, "subtotal": 536.36,
           "vat_rate": 10, "items": []}
    assert validators.arithmetic_check(bad)


def test_normalize_card():
    doc = validators.normalize({"card_last4": "**** 8920", "doc_date": "20/08/2026",
                                "doc_time": "22:37:57", "currency": None})
    assert doc["card_last4"] == "8920"
    assert doc["doc_date"] == "2026-08-20"
    assert doc["doc_time"] == "22:37"
    assert doc["currency"] == "TRY"


if __name__ == "__main__":
    test_vkn_checksum()
    test_date_normalize()
    test_arithmetic()
    test_normalize_card()
    print("Tüm testler geçti.")
