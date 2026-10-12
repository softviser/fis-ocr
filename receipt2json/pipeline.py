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
        ocr_text = self.ocr_checked(image_path)
        return self.structure_checked(ocr_text, keep_ocr=keep_ocr)

    def ocr_checked(self, image_path: str, model: str | None = None) -> str:
        """OCR + bozulma denetimi; model raydan çıkarsa yedek modelle dener.

        OCR modelleri bazı kırpma/en-boy oranlarında tekrarlı saçmalık
        üretebiliyor (ör. aynı satırı yüzlerce kez). Böyle çıktı tespit
        edilirse config'teki fallback_ocr_model ile bir kez daha denenir.
        """
        from .config import CFG

        text = self.ocr(image_path, model=model)
        if not _ocr_degenerate(text):
            return text
        fallback = CFG.get("fallback_ocr_model")
        used = model or self.ocr_model
        if fallback and fallback != used:
            retry = self.ocr(image_path, model=fallback)
            if not _ocr_degenerate(retry):
                return retry
        return text

    def structure_checked(self, ocr_text: str, keep_ocr: bool = False,
                          max_retries: int = 1) -> dict:
        """Yapılandırma + topraklama; model uydurursa yeniden dener.

        Yapılandırma modeli nadiren OCR metnini yok sayıp ezberindeki şablon
        bir faturayı üretebiliyor. Topraklama kontrolü 2+ 'OCR metninde yok'
        ihlali görürse aynı metinle bir kez daha denenir ve daha az ihlalli
        sonuç alınır.
        """
        doc = self.postprocess(self.structure(ocr_text), ocr_text, keep_ocr=keep_ocr)
        for attempt in range(max_retries):
            if _fabrication_count(doc) < 2:
                break
            retry = self.postprocess(
                self.structure(ocr_text), ocr_text, keep_ocr=keep_ocr
            )
            if _fabrication_count(retry) < _fabrication_count(doc):
                retry["_retried"] = attempt + 1
                doc = retry
        return doc

    def run_auto(
        self, image_path: str, mode: str = "two_stage", keep_ocr: bool = False
    ) -> list[dict]:
        """Görselde birden fazla fiş varsa (OpenCV kuruluysa) ayırıp her birini işle.

        Her zaman liste döner; tek fişte tek elemanlı liste.
        """
        from . import segment

        parts = segment.split_receipts(image_path)
        docs = [self.run(p, mode=mode, keep_ocr=keep_ocr) for p in parts]
        if len(docs) > 1:
            # Boş kırpmalardan (arka plan/boşluk) gelen içeriksiz sonuçları ele
            nonempty = [d for d in docs if not _is_empty_doc(d)]
            docs = nonempty or docs[:1]
        return docs


def _ocr_degenerate(text: str) -> bool:
    """OCR çıktısı bozuk mu? (çok kısa, tekrar döngüsü veya rakamsız)"""
    t = (text or "").strip()
    if len(t) < 40:
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.3:
        return True  # aynı satırlar döngüde tekrar ediyor
    if sum(c.isdigit() for c in t) < 10:
        return True  # fişte mutlaka rakam olur (tarih, tutar, no)
    return False


def _is_empty_doc(doc: dict) -> bool:
    """Kırpma fiş içermiyorsa model tüm kritik alanları boş döndürür."""
    return (
        not doc.get("vendor_tax_no")
        and doc.get("grand_total") is None
        and not doc.get("doc_no")
        and not doc.get("items")
    )


def _fabrication_count(doc: dict) -> int:
    """Uydurma işareti sayısı: OCR metninde bulunamayan kritik alan adedi."""
    return sum(
        1 for w in doc.get("_warnings", []) if "OCR metninde yok" in w
    )
