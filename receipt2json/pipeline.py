"""İki aşamalı fiş -> JSON pipeline'ı.

Aşama 1 (OCR):     deepseek-ocr / glm-ocr / llama3.2-vision -> ham metin
Aşama 2 (YAPI):    qwen3-coder:30b + JSON schema -> yapılandırılmış JSON
Aşama 3 (KONTROL): deterministik Python -> normalize, sağlama, aritmetik, eşleştirme
"""
import logging
import os

from . import prompts, validators
from .ollama_client import OllamaClient, parse_json_loose  # noqa: F401 - vlm_split kullanıyor
from .schema import RECEIPT_SCHEMA

log = logging.getLogger("receipt2json.pipeline")

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
        """Görselde birden fazla fiş varsa ayırıp her birini işle.

        Her zaman liste döner; tek fişte tek elemanlı liste.
        """
        from . import segment

        parts = segment.split_receipts(image_path)
        if len(parts) < 2 and segment.looks_multi(image_path):
            # Masaya serilmiş çok fiş: projeksiyon bölemedi ama sahne çok
            # fişli görünüyor. Koca görseli OCR'a sokmak çöp üretir; önce
            # VLM'den fiş kutularını isteyip tek tek kırpmak çok daha sağlam.
            vparts = self.vlm_split(image_path)
            if len(vparts) >= 2:
                parts = vparts
        docs: list[dict] = []
        for part in parts:
            docs.extend(self._process_part(part, mode, keep_ocr, depth=0))
        if len(docs) > 1:
            # Boş kırpmalardan (arka plan/boşluk) gelen içeriksiz sonuçları ele
            nonempty = [d for d in docs if not _is_empty_doc(d)]
            docs = nonempty or docs[:1]
        return docs

    def _process_part(self, part: str, mode: str, keep_ocr: bool,
                      depth: int) -> list[dict]:
        """Bir kırpmayı işle; hâlâ çok fişliyse (2+ TOPLAM kanıtı) sırayla
        satır bölme -> sütun bölme -> VLM kutu tespiti ile parçalayıp
        her parçayı özyinelemeli işle (en çok 2 seviye)."""
        from . import segment

        try:
            doc = self.run(part, mode=mode, keep_ocr=keep_ocr)
        except Exception as e:  # noqa: BLE001 - bir kırpmanın hatası (örn.
            # Ollama zaman aşımı) kalan fişlerin işlenmesini engellemesin
            log.error("Kırpma işlenemedi %s: %s", part, e)
            return [_error_doc(str(e))]
        # Bölmeyi tetikleyen durumlar:
        #  - 2+ TOPLAM kanıtı (her seviyede)
        #  - kök seviyede HER şüpheli sonuç: ızgara dizilimli çok fişli
        #    karelerde OCR çöpe döner veya model uydurur; TOPLAM sinyali
        #    hiç oluşmayabilir ama görseli bölmek sorunu çözer
        should_split = _multi_warned(doc) or (depth == 0 and doc.get("_suspect"))
        if mode != "two_stage" or not should_split or depth >= 2:
            return [doc]

        splitters = [segment.split_rows, segment.split_cols]
        if depth == 0:
            # Kırpma hâlâ masa sahnesi gibiyse VLM en güvenilir bölücü;
            # satır/sütun projeksiyonunun yarım bölmeleri onu gölgelemesin
            if segment.looks_multi(part):
                splitters.insert(0, self.vlm_split)
            else:
                splitters.append(self.vlm_split)
        for splitter in splitters:
            subs = splitter(part)
            if len(subs) < 2:
                continue
            subdocs: list[dict] = []
            for s in subs:
                subdocs.extend(self._process_part(s, mode, keep_ocr, depth + 1))
            subdocs = [d for d in subdocs if not _is_empty_doc(d)]
            if len(subdocs) >= 2:
                return subdocs
            if len(subdocs) == 1 and _is_empty_doc(doc):
                return subdocs
        return [doc]

    def vlm_split(self, image_path: str) -> list[str]:
        """Vision modelinden fiş sınırlayıcı kutuları isteyip kırp.

        Projeksiyon tabanlı bölme eğik/ızgara dizilimlerde (masaya serilmiş
        çok sayıda fiş) kör kalır; VLM kutuları bu durumu kurtarır.
        Koordinatlar 0-1000 normalize istenir; kutular `segment.crop_norm_boxes`
        ile doğrulanır (metinsiz/mükerrer/bozuk kutular elenir).
        """
        from . import segment

        schema = {
            "type": "object",
            "properties": {"receipts": {"type": "array", "items": {
                "type": "object",
                "properties": {"x0": {"type": "integer"}, "y0": {"type": "integer"},
                               "x1": {"type": "integer"}, "y1": {"type": "integer"}},
                "required": ["x0", "y0", "x1", "y1"],
            }}},
            "required": ["receipts"],
        }
        try:
            raw = self.client.chat(
                self.vision_model,
                "Bu fotoğrafta masaya serilmiş yazar kasa fişleri ve banka "
                "slipleri var. HER fiş/slip için ayrı bir sınırlayıcı kutu ver "
                "(yan yana ve alt alta duranlar dahil, hiçbirini atlama). "
                "Koordinatlar 0-1000 normalize: x0,y0 sol üst köşe; x1,y1 sağ "
                "alt köşe. Masa, gölge veya boş alan için kutu verme.",
                images=[image_path],
                format_schema=schema,
            )
            boxes = parse_json_loose(raw).get("receipts", [])
        except Exception:  # noqa: BLE001 - VLM yoksa/bozuksa sessizce vazgeç
            return [image_path]

        paths = segment.crop_norm_boxes(image_path, boxes)
        return paths if len(paths) >= 2 else [image_path]


def _error_doc(message: str) -> dict:
    """İşlenemeyen kırpma için işaretli boş sonuç (kardeş fişler etkilenmez)."""
    return {
        "vendor_name": None, "vendor_tax_no": None, "doc_date": None,
        "doc_no": None, "category": "diger", "payment_method": "bilinmiyor",
        "currency": "TRY", "items": [], "grand_total": None,
        "vat_amount": None, "notes": None,
        "_suspect": True,
        "_warnings": [f"İşleme hatası: {message}"],
    }


def _multi_warned(doc: dict) -> bool:
    return any(
        "birden fazla fiş olabilir" in w for w in doc.get("_warnings", [])
    )


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
