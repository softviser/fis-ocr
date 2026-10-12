# Fiş OCR

**Türk yazar kasa fişlerinin okunması ve JSON formatına çevrilmesi** için açık
kaynak bir araç. Fiş fotoğrafını verirsiniz; tamamen yerel çalışan Ollama
modelleriyle (hiçbir veri dışarı çıkmaz) fişi okur, yapılandırılmış JSON üretir,
vergi numarasını matematiksel olarak doğrular ve sonucu asenkron bir web
servisi + arşiv arayüzüyle sunar.

Desteklenen belgeler: yazar kasa fişi, e-arşiv fatura, POS slibi.

Bu proje [Softviser Yazılım](https://www.softviser.com.tr) tarafından
geliştirilmiştir ve MIT lisansıyla açık kaynaktır.

## Özellikler

- 📷 **Görsel → JSON**: firma, VKN, tarih, fiş no, kalemler, KDV, toplam, kart
  bilgileri, onay kodu... tam yapılandırılmış çıktı
- ✂️ **Çok fişli kare desteği**: bir fotoğrafta 2-3 fiş varsa otomatik tespit
  edilip kırpılır, her fiş ayrı işlenir
- ✅ **Deterministik doğrulama**: VKN/TCKN sağlama algoritması, KDV/toplam
  aritmetik kontrolü, halüsinasyon (topraklama) denetimi — model uydurursa
  çıktı `_suspect` olarak işaretlenir
- 🔁 **Asenkron iş kuyruğu**: yükleme anında kuyruk numarası döner, sonuç
  hazır olunca alınır; SQLite ile kalıcı, çökme sonrası işler devralınır
- 🖥️ **Web arayüzü**: sürükle-bırak / pano yapıştırma (Ctrl+V) ile yükleme,
  arşiv (arama, filtre, sıralama, silme), API dokümantasyonu
- 🧹 **Otomatik temizlik**: config'te tanımlı günden eski işler görselleriyle
  birlikte silinir
- 🔒 **Tamamen yerel**: Ollama + yerel modeller; bulut servisi yok

## Nasıl çalışır

```
Görsel ──► [0. Bölütleme]     OpenCV / saf Pillow (çok fişli kare → tek fişler)
       ──► [1. OCR]           deepseek-ocr                   ──► ham metin
       ──► [2. Yapılandırma]  qwen3-coder + JSON schema      ──► ham JSON
       ──► [3. Doğrulama]     deterministik Python           ──► nihai JSON
```

**Neden iki aşama?** Vision modeline tek seferde "görseli JSON'a çevir" demek
küçük modellerde tutar/numara alanlarında halüsinasyona açıktır. OCR'a özel
modeller metni sadık çıkarır; metin → JSON dönüşümünü yapılandırma modeli,
Ollama'nın **structured output** özelliğiyle (istek gövdesindeki `format`
alanına JSON şeması verilerek) şemaya %100 uyumlu yapar. (Not: Ollama yalnızca
şemada `required` olan alanları garanti eder — şemadaki tüm alanlar
required + nullable tanımlıdır.)

**3. aşama LLM'e bırakılmaz**, çünkü deterministik kontrol edilebilir:

- `vendor_tax_no` → VKN (10 hane) / TCKN (11 hane) **sağlama algoritması**
- `subtotal + vat_amount == grand_total` ve `vat ≈ grand × oran/(100+oran)`
- **Topraklama kontrolü**: modelin yazdığı kritik değerler (vergi no, fiş no,
  onay kodu, kart son4, toplamlar, firma adı) OCR metninde gerçekten geçmiyorsa
  `_suspect: true` + alan alan `_warnings`
- Tarih/saat/kart-son4 normalizasyonu, TR sayı formatı (`3.940,00` → `3940.0`)
- `data/vendors.json` ile kayıtlı firma eşleştirme → `_tax_check`
  (sağlaması bozuk numarada 1 karakter farklı kayıtlı VKN önerilir)
- `data/cards.json` ile kart eşleştirme → `_card_hint`

**Güven kuralı:** `_suspect` yok + `_warnings` boş + `_tax_check.valid` →
otomatik onaylanabilir; aksi halde insan kontrol kuyruğu.

## Model karşılaştırması ve test sonuçları

Test ortamı: Ollama 0.40.3 (uzak GPU sunucu), gerçek Türk fiş fotoğrafları.
Tarih: 2026-10-12. Kendi fişlerinizle tekrarlamak için:
`python benchmark.py fis.jpg`

### Model karşılaştırması (10 kritik alan üzerinden)

Kritik alanlar: firma, VKN, tarih, saat, fiş no, toplam, KDV, kart son4,
onay kodu, ödeme türü. İki görsel: temiz fiş ve buruşuk/soluk fiş.

| Yol | Temiz fiş | Buruşuk fiş | Süre/görsel | Not |
|---|---|---|---|---|
| **deepseek-ocr + qwen3-coder:30b** (varsayılan) | **10/10** | **10/10** | **33-40 sn** | En hızlı iki aşamalı yol; uyarısız temiz çıktı |
| glm-ocr + qwen3-coder:30b | 10/10 | 10/10 ¹ | 47 sn | Alanlar doğru ama OCR metninde tekrarlar ürettiği için yanlış-pozitif `_suspect` uyarıları tetikledi |
| minicpm-v4.5 + qwen3-coder:30b | 10/10 | 10/10 ¹ | 39-59 sn | Temiz ve doğru; deepseek'e güçlü alternatif |
| qwen3-vl:8b + qwen3-coder:30b | 10/10 | 10/10 ¹ | 63-69 sn | Doğru ama en yavaş iki aşamalı yol |
| qwen3-vl:8b (tek aşamalı direkt) | 10/10 | **9/10** ² | **25-54 sn** | Temiz fişte en hızlı; buruşukta onay kodunun 1 hanesini uydurdu |
| llama3.2-vision (tek aşamalı) | çalışmadı ³ | — | — | — |

¹ Firma olarak tabela adı (ESPRESSOLAB) çıkarıldı; deepseek-ocr yasal unvanı
(PARKON GIDA İNŞAAT TİC.LTD.ŞTİ.) çıkardı — ikisi de fişte yazıyor, ikisi de doğru.

² `qwen3-vl:8b` direkt modda buruşuk fişte onay kodunu `718745` okudu (doğrusu
`710745`) — **tek aşamalı modun bilinen riski**: OCR metni olmadığı için
topraklama kontrolü devreye giremez ve uydurulan hane sessizce geçer. İki
aşamalı yolun ve topraklama denetiminin varlık sebebi tam olarak budur.

³ `llama3.2-vision` bu sunucuda yeniden indirildikten sonra bile Ollama 500
döndürdü (şemasız basit istekte dahi; diğer 5 model aynı anda sorunsuz).
Ollama 0.40.3 ile uyumsuzluk görünüyor; `ollama rm llama3.2-vision` ile
kaldırılabilir, yeri `qwen3-vl:8b` (config → `vision_model`) alır.

**Sonuç:** dört OCR yolu da alan doğruluğunda eşit; **`deepseek-ocr` hız ve
uyarı temizliğiyle varsayılan kalıyor**. `minicpm-v4.5` en güçlü yedek.
`qwen3-vl:8b` hızlı ön izleme/direkt mod için uygun ama kritik (muhasebe)
akışında iki aşamalı yol + topraklama şart.

### Uçtan uca sistem testleri (gerçek fişlerle)

| Senaryo | Sonuç |
|---|---|
| Temiz tek fiş | Tüm alanlar doğru; VKN sağlaması ✓, kayıtlı firma eşleşti |
| Buruşuk/soluk fiş | Kalemler, kart bilgileri, onay kodu, saat, kasiyer dahi doğru |
| Çok fişli kare bölütleme (5 görsel: 2+3+1+1+2 fiş) | **5/5 doğru parça sayısı**; tek fişliler hiç yanlış bölünmedi |
| 3 fişli kare, uçtan uca | 3 ayrı JSON (98 sn); fiziksel örtüşen 1 kırpma karıştı → topraklama kontrolü `_suspect` ile yakaladı |
| OCR tamamen çökünce model uydurma fatura üretti | Topraklama kontrolü 7 ihlalle `_suspect: true` işaretledi — uydurma çıktı sessizce geçemiyor |
| VKN sağlama algoritması | Gerçek VKN'lerle doğrulandı; 1 hane bozuk numarayı reddediyor |

## Yapılandırma — config.json

Tüm ayarlar proje kökündeki **`config.json`** dosyasından okunur
(değişiklik sonrası servisi yeniden başlatın):

```json
{
    "ollama_host": "http://127.0.0.1:11434",
    "ocr_model": "deepseek-ocr:latest",
    "structure_model": "qwen3-coder:30b",
    "vision_model": "llama3.2-vision:latest",
    "max_upload_mb": 20,
    "worker_count": 1,
    "data_dir": "service_data",
    "retention_days": 30,
    "cleanup_interval_hours": 6
}
```

- **`max_upload_mb`** — yükleme sınırı; hata mesajları ve upload sayfası da
  buradan beslenir.
- **`retention_days`** — bu kadar günden eski *bitmiş* işler (kayıt + görsel)
  otomatik silinir; `0`/`null` = kapalı. Kontrol sıklığı `cleanup_interval_hours`.
- **`data_dir`** — göreli yol proje köküne göredir.
- Öncelik: **ortam değişkeni → config.local.json → config.json → varsayılan**
  (`OLLAMA_HOST`, `WORKER_COUNT`, `RECEIPT2JSON_DATA`; config dosyası yolu
  `RECEIPT2JSON_CONFIG` ile değiştirilebilir).

### Gerçek bilgiler: `*.local.json` düzeni

Repoya özel/gerçek bilgi girmemek için her config/veri dosyasının yanında
**`.local.json`** uzantılı, `.gitignore`'da olan bir eş desteklenir; varsa
ana dosyanın üstüne birleştirilir:

| Repoya giren (örnek değerler) | Lokal/sunucuya özel (git'e girmez) |
|---|---|
| `config.json` | `config.local.json` — gerçek Ollama adresi vb. |
| `data/vendors.json` | `data/vendors.local.json` — gerçek firma kayıtları |
| `data/cards.json` | `data/cards.local.json` — gerçek kart tanımları |

## Gereksinimler

- Python 3.10+
- Erişilebilir bir [Ollama](https://ollama.com) sunucusu ve modeller:

```bash
ollama pull deepseek-ocr
ollama pull qwen3-coder:30b
ollama pull qwen3-vl:8b       # yalnız 'direct' mod için, isteğe bağlı
```

## Ubuntu sunucuya kurulum (tek script)

Projeyi sunucuya kopyalayın (`scp -r` veya `git clone`), sonra:

```bash
# Kur + 8080 portunda ön planda başlat (Python/venv/pip eksikse apt ile kurar)
sudo bash install.sh

# Farklı port / farklı Ollama adresi
sudo bash install.sh --port 9090 --ollama http://10.0.0.100:11434

# Kalıcı kurulum: systemd servisi (açılışta başlar, çökerse yeniden kalkar)
sudo bash install.sh --service --port 8080

# Kaldırma
sudo bash install.sh --remove           # servis + venv kaldır, veriler kalır
sudo bash install.sh --remove --purge   # veriler (service_data/) dahil sil
```

Servis yönetimi:

```bash
journalctl -u fis-ocr -f        # loglar
systemctl restart fis-ocr       # yeniden başlat
systemctl stop fis-ocr          # durdur
```

**Güncelleme (çalışan sisteme yeni sürüm):** yeni kodu aynı klasörün üzerine
kopyalayın (`service_data/` ve özelleştirdiyseniz `config.json`'a dokunmayın) →
`sudo bash install.sh --service`. Script birimi yeniden yazar, bağımlılıkları
günceller ve eski servisi kapatıp yenisini başlatır.

Klasör konumu/adı önemli değildir (`/opt/fis-ocr` önerilir; yolda boşluk
olmasın). Taşıma sonrası `--service` komutunu bir kez daha çalıştırın.

## Kullanım (CLI)

```bash
pip install -r requirements.txt

python -m receipt2json.cli fis.jpg                  # tek fiş, stdout'a
python -m receipt2json.cli fisler/ --out cikti/     # klasör toplu işleme
python -m receipt2json.cli fis.jpg --mode direct    # tek aşamalı vision modu
python -m receipt2json.cli fis.jpg --ocr-model glm-ocr:latest
python -m receipt2json.cli fis.jpg --keep-ocr       # ham OCR metnini sakla
```

## Web servisi

```bash
# Sunucuda (FastAPI):
uvicorn receipt2json.service.app:app --host 0.0.0.0 --port 8080

# FastAPI kurulamayan makinede saf stdlib sürüm — aynı API:
python -m receipt2json.service.stdlib_app --port 8077
```

### Sayfalar

| Yol | İçerik |
|---|---|
| `/` | Upload: sürükle-bırak, **Ctrl+V pano yapıştırma**, kuyruk takibi, sonuç kartları |
| `/arsiv` | Arşiv: arama, durum filtresi, tarih/tutar sıralama, silme, görsel + JSON detayı |
| `/docs` | API dokümantasyonu (FastAPI'de `/api-docs`; Swagger `/docs`) |

### API özeti

```bash
# İş gönder (anında 202 + job_id döner)
curl -F "file=@fis.jpg" http://SUNUCU:8080/api/jobs

# Base64 / pano görseli ile gönder
curl -X POST http://SUNUCU:8080/api/jobs/base64 \
     -H "Content-Type: application/json" \
     -d '{"filename":"fis.jpg","data":"<base64 veya data-URL>"}'

# Sonuç sorgula (queued -> processing -> done|error)
curl http://SUNUCU:8080/api/jobs/<job_id>

# Arşiv, görsel, silme, istatistik, sağlık
curl "http://SUNUCU:8080/api/jobs?status=done&q=KEBAP&limit=100"
curl http://SUNUCU:8080/api/jobs/<job_id>/image -o fis.jpg
curl -X DELETE http://SUNUCU:8080/api/jobs/<job_id>
curl http://SUNUCU:8080/api/stats
curl http://SUNUCU:8080/health
```

Çok fişli karede `result`: `{"_multi": true, "receipt_count": 3, "receipts": [...]}`.
Ayrıntılı doküman ve örnek kodlar (PowerShell / Python / JS): servis üzerindeki `/docs`.

## Proje yapısı

```
fis-ocr/
├── config.json              # tüm ayarlar
├── install.sh               # Ubuntu kurulum/servis/kaldırma scripti
├── requirements.txt
├── benchmark.py             # OCR model karşılaştırma aracı
├── LICENSE                  # MIT
├── data/                    # kayıtlı firma/kart eşleştirme veritabanları
│   ├── vendors.json
│   └── cards.json
├── receipt2json/            # ana paket
│   ├── config.py            # config.json yükleyici
│   ├── pipeline.py          # bölütleme → OCR → yapılandırma → doğrulama
│   ├── segment.py           # çok fişli kare kırpma (OpenCV / saf Pillow)
│   ├── validators.py        # VKN/TCKN, aritmetik, topraklama
│   ├── ollama_client.py · prompts.py · schema.py · cli.py
│   └── service/             # web servisi
│       ├── app.py           # FastAPI (sunucu)
│       ├── stdlib_app.py    # saf stdlib sürüm (kısıtlı makineler)
│       ├── store.py         # SQLite kuyruk + arşiv + otomatik temizlik
│       ├── worker.py        # worker havuzu + RetentionCleaner
│       └── *.html           # upload / arşiv / API doküman sayfaları
├── tests/                   # Ollama gerektirmeyen testler
└── service_data/            # çalışma verisi (git'e girmez)
```

## Testler

```bash
python tests/test_validators.py    # VKN/TCKN sağlama, aritmetik, normalizasyon
python tests/test_queue_core.py    # kuyruk + worker + crash-recovery + otomatik temizlik
python tests/test_service.py       # HTTP API uçtan uca (FastAPI, FAKE mod)
```

Testler Ollama gerektirmez (`RECEIPT2JSON_FAKE=1` sahte modu kullanılır).

## Katkı

Hata bildirimi ve katkılarınızı bekleriz. Yeni fiş formatı desteği eklerken
`receipt2json/prompts.py` içindeki alan eşleme kurallarına ve
`tests/`'e örnek eklemeniz yeterlidir.

## Lisans

MIT — ayrıntı için [LICENSE](LICENSE). © 2026 [Softviser Yazılım](https://www.softviser.com.tr)
