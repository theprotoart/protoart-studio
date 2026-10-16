#!/bin/bash
# One-time setup for Protoart Studio: Python environment + Electron app.
# Safe to run again (e.g. to update diffusers).
set -e
cd "$(dirname "$0")"

echo "== Protoart Studio setup =="

if [ "$(uname -m)" != "arm64" ]; then
  echo "Warning: this Mac isn't Apple Silicon. Generation will run on the CPU and be very slow."
fi

FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
if [ "${FREE_GB:-0}" -lt 6 ]; then
  echo "Only ${FREE_GB} GB free on this disk. Setup needs about 6 GB (models need more later)."
  echo "Free up space and run setup.sh again."
  exit 1
fi

# Find Python 3.10 or newer.
PY=""
for c in python3.12 python3.11 python3.13 python3.10 "$HOME/miniconda3/bin/python3" /opt/homebrew/bin/python3 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10+ not found. Install it with:  brew install python@3.12   (or from python.org)"
  exit 1
fi
echo "Using $($PY --version) at $(command -v $PY)"

# Check that Python can reach PyPI before installing anything.
if ! "$PY" -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/pip/', timeout=15)" 2>/tmp/protoart_net.txt; then
  echo
  echo "Python can't connect to pypi.org (needed to download packages)."
  echo "Error: $(tail -1 /tmp/protoart_net.txt)"
  if env | grep -qi '_proxy='; then
    echo "A proxy is set in this Terminal:"; env | grep -i '_proxy='
    echo "If you don't use a proxy, run:  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY"
  fi
  if curl -s -o /dev/null https://pypi.org/simple/pip/; then
    echo "curl CAN connect, so something is blocking Python specifically:"
    echo "  - a firewall/filter app (Little Snitch, LuLu, VPN): allow Python, or pause it"
    echo "  - or update Python:  brew upgrade python@3.12"
  else
    echo "Nothing can connect: check your internet, VPN or firewall."
  fi
  exit 1
fi

if [ ! -x backend/.venv/bin/python ]; then
  "$PY" -m venv backend/.venv
fi
backend/.venv/bin/python -m pip install --upgrade pip
# Exact tested versions (requirements.lock.txt); requirements.txt lists the top-level packages.
backend/.venv/bin/python -m pip install -r backend/requirements.lock.txt

echo "== Checking the Python side =="
backend/.venv/bin/python - <<'EOF'
import torch, diffusers
print("torch", torch.__version__, "| Apple GPU (MPS):", torch.backends.mps.is_available())
print("diffusers", diffusers.__version__, "| ERNIE support:", hasattr(diffusers, "ErnieImagePipeline"))
EOF
backend/.venv/bin/python -m unittest discover -s backend/tests -t backend

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found. Install Node.js (brew install node, or nodejs.org), then run setup.sh again."
  exit 1
fi
(cd app && npm install)
# Newer npm skips install scripts by default, so Electron's own download step may not have run.
if [ ! -f app/node_modules/electron/path.txt ]; then
  echo "Downloading the Electron app runtime…"
  (cd app && node node_modules/electron/install.js)
fi

DATA="$HOME/.protoartstudio"; [ -d "$DATA" ] || [ ! -d "$HOME/.beestudio" ] || DATA="$HOME/.beestudio"
mkdir -p "$DATA/loras" "$DATA/models"
echo
echo "Done. Start Protoart Studio with:  ./start.sh"
echo "Put LoRA .safetensors files in: $DATA/loras"
