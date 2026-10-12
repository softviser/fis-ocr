#!/usr/bin/env bash
# =============================================================================
# NV-Stok Fiş OCR — Ubuntu kurulum ve çalıştırma scripti
#
# Kullanım:
#   sudo bash install.sh                        # kur + 8080 portunda başlat
#   sudo bash install.sh --port 9090            # farklı portta başlat
#   sudo bash install.sh --service              # systemd servisi olarak kur
#                                               # (açılışta otomatik başlar)
#   sudo bash install.sh --service --port 9090
#   bash install.sh --run                       # kurulum zaten yapıldıysa
#                                               # sadece çalıştır
#
#   sudo bash install.sh --remove               # servisi kaldır (veriler kalır)
#   sudo bash install.sh --remove --purge       # servis + veriler + venv sil
#
# Seçenekler:
#   --port <N>     Dinlenecek port (varsayılan: 8080; PORT env ile de verilebilir)
#   --ollama <URL> Ollama adresi (varsayılan: http://127.0.0.1:11434;
#                  ayrı makinedeyse örn. http://10.0.0.100:11434)
#   --service      systemd servisi kur/güncelle ve başlat (fis-ocr.service)
#   --run          Kurulumu atla, ön planda çalıştır
#   --no-start     Yalnızca kur, başlatma
#   --remove       Servisi durdur, devre dışı bırak ve birim dosyasını sil;
#                  .venv silinir, veriler (service_data/) ve kod KALIR
#   --purge        --remove ile birlikte: service_data/ (yüklenen görseller +
#                  iş veritabanı) da kalıcı olarak silinir
# =============================================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$APP_DIR/.venv"
SERVICE_NAME="fis-ocr"

PORT_SET=0; [[ -n "${PORT:-}" ]] && PORT_SET=1
OLLAMA_SET=0; [[ -n "${OLLAMA_HOST:-}" ]] && OLLAMA_SET=1
PORT="${PORT:-8080}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
AS_SERVICE=0
RUN_ONLY=0
NO_START=0
REMOVE=0
PURGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)    PORT="$2"; PORT_SET=1; shift 2 ;;
    --ollama)  OLLAMA_HOST="$2"; OLLAMA_SET=1; shift 2 ;;
    --service) AS_SERVICE=1; shift ;;
    --run)     RUN_ONLY=1; shift ;;
    --no-start) NO_START=1; shift ;;
    --remove)  REMOVE=1; shift ;;
    --purge)   PURGE=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Bilinmeyen parametre: $1 (yardım: --help)"; exit 1 ;;
  esac
done

log()  { echo -e "\e[1;32m[kurulum]\e[0m $*"; }
warn() { echo -e "\e[1;33m[uyarı]\e[0m $*"; }

# ---------------------------------------------------------------- kaldırma

if [[ "$REMOVE" -eq 1 ]]; then
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Kaldırma için root gerekli: sudo bash install.sh --remove"; exit 1
  fi
  if systemctl list-unit-files 2>/dev/null | grep -q "^$SERVICE_NAME.service"; then
    log "Servis durduruluyor ve kaldırılıyor: $SERVICE_NAME"
    systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true
    rm -f "/etc/systemd/system/$SERVICE_NAME.service"
    systemctl daemon-reload
  else
    log "Kurulu $SERVICE_NAME servisi bulunamadı; dosyalar temizleniyor."
  fi
  if [[ -d "$VENV" ]]; then
    rm -rf "$VENV"
    log "Sanal ortam silindi: $VENV"
  fi
  if [[ "$PURGE" -eq 1 ]]; then
    if [[ -d "$APP_DIR/service_data" ]]; then
      rm -rf "$APP_DIR/service_data"
      log "Veriler silindi: $APP_DIR/service_data (görseller + iş veritabanı)"
    fi
  else
    log "Veriler korundu: $APP_DIR/service_data (silmek için: --remove --purge)"
  fi
  log "Kaldırma tamamlandı. Proje kodu yerinde duruyor: $APP_DIR"
  exit 0
fi

if [[ "$PURGE" -eq 1 ]]; then
  echo "--purge yalnızca --remove ile kullanılır."; exit 1
fi

# ---------------------------------------------------------------- interaktif

# Parametre verilmediyse ve terminal etkileşimliyse sor
if [[ -t 0 ]]; then
  if [[ "$PORT_SET" -eq 0 ]]; then
    read -rp "Web servis hangi portta yayınlansın? [8080]: " ans
    PORT="${ans:-8080}"
  fi
  if [[ "$OLLAMA_SET" -eq 0 ]]; then
    echo "Ollama sunucu adresi (bu makinedeyse Enter'a basın):"
    read -rp "  [http://127.0.0.1:11434]: " ans
    OLLAMA_HOST="${ans:-http://127.0.0.1:11434}"
  fi
fi

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Geçersiz port: $PORT"; exit 1
fi

# ---------------------------------------------------------------- 1. sistem

if [[ "$RUN_ONLY" -eq 0 ]]; then
  if ! command -v python3 >/dev/null 2>&1 \
     || ! python3 -c 'import venv' >/dev/null 2>&1 \
     || ! command -v pip3 >/dev/null 2>&1; then
    log "Python / venv / pip eksik; apt ile kuruluyor..."
    if [[ "$(id -u)" -ne 0 ]]; then
      echo "Paket kurulumu için root gerekli: sudo bash install.sh"; exit 1
    fi
    apt-get update -y
    apt-get install -y python3 python3-venv python3-pip curl
  else
    log "Python mevcut: $(python3 --version)"
  fi

  # ---------------------------------------------------------------- 2. venv

  if [[ ! -d "$VENV" ]]; then
    log "Sanal ortam oluşturuluyor: $VENV"
    python3 -m venv "$VENV"
  fi
  log "Bağımlılıklar yükleniyor..."
  "$VENV/bin/pip" install --upgrade pip -q
  "$VENV/bin/pip" install -q -r "$APP_DIR/requirements.txt"
  log "Bağımlılıklar tamam."

  # ---------------------------------------------------------------- 3. ollama kontrolü

  if command -v curl >/dev/null 2>&1; then
    if curl -sf -m 5 "$OLLAMA_HOST/api/tags" >/dev/null; then
      log "Ollama erişilebilir: $OLLAMA_HOST"
    else
      warn "Ollama'ya erişilemedi: $OLLAMA_HOST"
      warn "Servis yine de kurulur; Ollama ayağa kalkınca işler işlenir."
      warn "Farklı adres için: --ollama http://IP:11434"
    fi
  fi
fi

if [[ ! -x "$VENV/bin/uvicorn" ]]; then
  echo "Kurulum eksik görünüyor ($VENV yok). Önce: sudo bash install.sh"; exit 1
fi

# ---------------------------------------------------------------- 4. çalıştırma

UVICORN_CMD=("$VENV/bin/uvicorn" "receipt2json.service.app:app"
             "--host" "0.0.0.0" "--port" "$PORT")

if [[ "$AS_SERVICE" -eq 1 ]]; then
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "systemd servisi kurmak için root gerekli: sudo bash install.sh --service"
    exit 1
  fi
  RUN_USER="${SUDO_USER:-root}"
  log "systemd servisi yazılıyor: /etc/systemd/system/$SERVICE_NAME.service"
  cat > "/etc/systemd/system/$SERVICE_NAME.service" <<EOF
[Unit]
Description=NV-Stok Fis OCR servisi (port $PORT)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$APP_DIR
Environment=OLLAMA_HOST=$OLLAMA_HOST
Environment=RECEIPT2JSON_DATA=$APP_DIR/service_data
ExecStart=${UVICORN_CMD[*]}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable "$SERVICE_NAME" >/dev/null
  if [[ "$NO_START" -eq 0 ]]; then
    systemctl restart "$SERVICE_NAME"
    sleep 2
    systemctl --no-pager --lines=5 status "$SERVICE_NAME" || true
  fi
  IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  log "Servis kuruldu. Arayüz: http://${IP:-SUNUCU_IP}:$PORT"
  log "Loglar:  journalctl -u $SERVICE_NAME -f"
  log "Durdur:  systemctl stop $SERVICE_NAME"
elif [[ "$NO_START" -eq 0 ]]; then
  IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  log "Ön planda başlatılıyor (kalıcı kurulum için: --service)"
  log "Arayüz: http://${IP:-SUNUCU_IP}:$PORT  —  durdurmak için Ctrl+C"
  cd "$APP_DIR"
  OLLAMA_HOST="$OLLAMA_HOST" RECEIPT2JSON_DATA="$APP_DIR/service_data" \
    exec "${UVICORN_CMD[@]}"
else
  log "Kurulum tamam (başlatılmadı). Çalıştır: bash install.sh --run --port $PORT"
fi
