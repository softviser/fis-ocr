"""Pipeline prompt'ları."""

OCR_PROMPT = (
    "Extract ALL text from this Turkish retail receipt image. "
    "Preserve the line order and layout. Output plain text only, "
    "do not translate, do not summarize, do not add commentary."
)

STRUCTURE_SYSTEM = """Sen Türk perakende fişlerini (yazar kasa fişi, e-arşiv fatura, POS slibi) \
yapılandırılmış JSON'a çeviren bir veri çıkarma asistanısın. Sana OCR ile çıkarılmış ham fiş \
metni verilecek. SADECE metinde gerçekten yazan bilgiyi kullan; emin olmadığın alanı null bırak. \
Asla uydurma."""

STRUCTURE_PROMPT = """Aşağıdaki OCR metnini verilen JSON şemasına dönüştür.

Alan eşleme kuralları (Türk fiş terminolojisi):
- Fişin en üstündeki büyük isim -> vendor_name. Hemen altındaki kişi/şirket satırları ve adres -> vendor_address.
- "... V.D." / "... VD" / "VERGİ DAİRESİ" yanındaki isim -> vendor_tax_office; yanındaki 10-11 haneli sayı -> vendor_tax_no.
- TARİH (GG/AA/YYYY) -> doc_date (YYYY-MM-DD formatına çevir). SAAT -> doc_time (HH:MM).
- FİŞ NO -> doc_no. (E-arşiv faturada Belge No -> doc_no.)
- category: klasik yazar kasa fişi ise "yazar_kasa", E-ARŞİV FATURA başlığı varsa "e_arsiv", sadece POS slibi ise "pos_slip", emin değilsen "diger".
- Satırlardaki ürünler -> items. "(2x) ADANA" gibi önekte adet varsa qty=2 ve unit_price = total/qty. Satır formatı genelde: AD %KDV *TUTAR. Ürün dökümü yoksa (örn. sadece "YİYECEK %10 *590,00") items'a tek kalem olarak yaz (name="YİYECEK", qty=1).
- TOPKDV -> vat_amount. TOPLAM -> grand_total. subtotal = grand_total - vat_amount (metinde ayrıca yazmıyorsa hesapla, hesaplayamıyorsan null).
- vat_rate: tüm kalemler aynı orandaysa o oran (%10 -> 10), karışıksa null.
- KREDİ -> payment_method="kredi_karti"; NAKİT -> "nakit".
- KREDİ satırının altındaki banka adı (örn. KUVEYT TÜRK, HALKBANK, İŞ BANKASI) -> card_bank.
- "**** **** **** 1234" veya "***********1234" -> card_last4="1234".
- "Visa payWave", "VISA", "MasterCard", "Troy" gibi ibareler -> card_type (VISA/MASTERCARD/TROY).
- ONAY KODU -> approval_code. BATCH NO -> batch_no.
- "Kullanıcı:" / kasiyer adı -> cashier.
- currency her zaman "TRY" (aksi açıkça yazmadıkça).
- Sayı formatı: Türk formatındaki "3.940,00" -> 3940.0 olarak yaz. Para alanları number tipinde.
- notes: Z NO, EKÜ NO, MF numarası, İŞYERİ NO, İŞLEM NO, POS NO, ÇEK/MASA NO, AID, temassız işlem bilgisi, e-arşiv ETTN/plaka/park süresi gibi şemaya girmeyen ama kayda değer her şeyi tek string'te topla ("; " ile ayır).

OCR METNİ:
---
{ocr_text}
---

Şemaya uygun JSON üret."""

# Tek aşamalı mod (vision modeli doğrudan JSON üretir) için
VISION_DIRECT_PROMPT = (
    "Bu bir Türk perakende fişi fotoğrafı. Fişteki tüm bilgileri oku ve "
    "verilen JSON şemasına uygun şekilde çıkar. Sadece görselde gerçekten "
    "yazan bilgiyi kullan, emin olmadığın alanı null bırak.\n\n"
    + STRUCTURE_PROMPT.split("OCR METNİ:")[0]
)
