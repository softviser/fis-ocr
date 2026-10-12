"""İki aşamalı fiş -> JSON pipeline'ı.

Aşama 1 (OCR):     deepseek-ocr / glm-ocr / llama3.2-vision -> ham metin
Aşama 2 (YAPI):    qwen3-coder:30b + JSON schema -> yapılandırılmış JSON
Aşama 3 (KONTROL): deterministik Python -> normalize, sağlama, aritmetik, eşleştirme
"""
import os

from . import prompts, validators
from .ollama_client import OllamaClient, parse_json_loose
from .schema import RECEIPT_SCHEMA

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
VENDORS_DB = os.path.join(DATA_DIR, "vendors.json")
CARDS_DB = os.path.join(DATA_DIR, "cards.json")


class ReceiptPipeline:
    def __init__(
        self,
        host: str | None = None,
        ocr_model: str | None = None,
        structure_model: str | None = None,
        vision_model: str | None = None,
    ):
        from .config import CFG

        self.client = OllamaClient(host)
        self.ocr_model = ocr_model or CFG["ocr_model"]
        self.structure_model = structure_model or CFG["structure_model"]
        self.vision_model = vision_model or CFG["vision_model"]

    # ------------------------------------------------------------ aşamalar

    def ocr(self, image_path: str, model: str | None = None) -> str:
        return self.client.chat(
            model=model or self.ocr_model,
            prompt=prompts.OCR_PROMPT,
            images=[image_path],
        )

    def structure(self, ocr_text: str) -> dict:
        raw = self.client.chat(
            model=self.structure_model,
            system=prompts.STRUCTURE_SYSTEM,
            prompt=prompts.STRUCTURE_PROMPT.format(ocr_text=ocr_text),
            format_schema=RECEIPT_SCHEMA,
            num_ctx=16384,
        )
        return parse_json_loose(raw)

    def extract_direct(self, image_path: str) -> dict:
        """Tek aşamalı mod: vision modeli doğrudan JSON üretir (karşılaştırma için)."""
        raw = self.client.chat(
            model=self.vision_model,
            prompt=prompts.VISION_DIRECT_PROMPT,
            images=[image_path],
            format_schema=RECEIPT_SCHEMA,
        )
        return parse_json_loose(raw)

    def postprocess(
        self, doc: dict, ocr_text: str | None = None, keep_ocr: bool = False
    ) -> dict:
        doc = validators.normalize(doc)
        problems = validators.arithmetic_check(doc)
        if ocr_text is not None:
            ground_warnings, suspect = validators.groundedness_check(doc, ocr_text)
            problems.extend(ground_warnings)
            if suspect:
                doc["_suspect"] = True
        doc["_tax_check"] = validators.tax_check(doc, VENDORS_DB)
        hint = validators.card_hint(doc, CARDS_DB)
        if hint:
            doc["_card_hint"] = hint
        if problems:
            doc["_warnings"] = problems
        if keep_ocr and ocr_text is not None:
            doc["_ocr_text"] = ocr_text
        return doc

    # ------------------------------------------------------------ uçtan uca

    def run(self, image_path: str, mode: str = "two_stage", keep_ocr: bool = False) -> dict:
        if mode == "direct":
            doc = self.extract_direct(image_path)
            return self.postprocess(doc)
        ocr_text = self.ocr(image_path)
        doc = self.structure(ocr_text)
        return self.postprocess(doc, ocr_text, keep_ocr=keep_ocr)

    def run_auto(
        self, image_path: str, mode: str = "two_stage", keep_ocr: bool = False
    ) -> list[dict]:
        """Görselde birden fazla fiş varsa (OpenCV kuruluysa) ayırıp her birini işle.

        Her zaman liste döner; tek fişte tek elemanlı liste.
        """
        from . import segment

        parts = segment.split_receipts(image_path)
        return [self.run(p, mode=mode, keep_ocr=keep_ocr) for p in parts]
