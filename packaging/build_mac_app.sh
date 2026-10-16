#!/bin/bash
# Builds "Protoart Studio.app": a standalone Mac app with its own Python inside.
#
# Reuses what's already on this Mac (the Python packages in backend/.venv and the Electron
# in app/node_modules), so it only downloads one small file (a portable Python, ~17 MB) the
# first time. On APFS disks the big files are cloned, not copied, so the app takes almost
# no extra disk space until something changes.
#
# Run it with "Build Protoart Studio.command" (double-click) or:  bash packaging/build_mac_app.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
APP_NAME="Protoart Studio"
BUNDLE_ID="com.protoart.studio"
OUT="$ROOT/dist"
APP="$OUT/$APP_NAME.app"
RES="$APP/Contents/Resources"
PY_VER="3.12.7"
PBS_TAG="20241016"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/cpython-${PY_VER}%2B${PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz"
# SHA-256 published by python-build-standalone for this exact file; the build stops if it doesn't match
PY_SHA256="4c18852bf9c1a11b56f21bcf0df1946f7e98ee43e9e4c0c5374b2b3765cf9508"
DATA="${PROTOART_HOME:-$HOME/.protoartstudio}"
CACHE="$DATA/build-cache"
VENV_SP="$ROOT/backend/.venv/lib/python3.12/site-packages"
ELECTRON_APP="$ROOT/app/node_modules/electron/dist/Electron.app"
VERSION="$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' "$ROOT/app/package.json" | head -1)"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mStopped: %s\033[0m\n' "$*"; exit 1; }
# clone on APFS (instant, no extra space), plain copy elsewhere
clone() { cp -cR "$1" "$2" 2>/dev/null || cp -R "$1" "$2"; }

step "Checking this Mac"
[ "$(uname -s)" = "Darwin" ] || fail "this builds a Mac app, so it has to run on a Mac."
[ "$(uname -m)" = "arm64" ] || fail "the standalone app is for Apple Silicon Macs (M1 or newer)."
[ -d "$VENV_SP" ] || fail "backend/.venv isn't set up. Run ./setup.sh first."
[ -d "$ELECTRON_APP" ] || fail "Electron isn't installed in app/node_modules. Run ./setup.sh first."
[ -f "$ROOT/app/node_modules/vue/dist/vue.global.prod.js" ] || fail "Vue is missing from app/node_modules. Run ./setup.sh first."
WANT_ELECTRON="$(sed -n 's/.*"electron": *"\([0-9.]*\)".*/\1/p' "$ROOT/app/package.json" | head -1)"
HAVE_ELECTRON="$(cat "$ROOT/app/node_modules/electron/dist/version" 2>/dev/null | tr -d 'v ')"
[ "$WANT_ELECTRON" = "$HAVE_ELECTRON" ] || fail "Electron ${HAVE_ELECTRON:-?} is installed but this version needs $WANT_ELECTRON. Run ./start.sh once (it updates Electron), then build again."
FREE_GB=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
echo "Free disk space: ${FREE_GB} GB · version ${VERSION}"
[ "${FREE_GB:-0}" -ge 2 ] || fail "less than 2 GB free. Free up some space first."

step "1/7 Portable Python ${PY_VER}"
mkdir -p "$CACHE" "$OUT"
TAR="$CACHE/cpython-${PY_VER}-${PBS_TAG}-aarch64-apple-darwin.tar.gz"
if [ ! -s "$TAR" ]; then
  echo "Downloading (one time, ~17 MB)…"
  curl -fL --retry 3 --progress-bar -o "$TAR.part" "$PY_URL" || fail "couldn't download Python. Check the internet connection and run this again."
  mv "$TAR.part" "$TAR"
else
  echo "Using the copy saved earlier."
fi
GOT_SHA="$(shasum -a 256 "$TAR" | cut -d' ' -f1)"
if [ "$GOT_SHA" != "$PY_SHA256" ]; then
  rm -f "$TAR"
  fail "the Python download doesn't match its published checksum, so it wasn't used (deleted; run again to re-download)."
fi
echo "Checksum verified."

step "2/7 App shell (Electron)"
if [ -e "$APP" ]; then
  echo "Replacing the earlier build in dist/…"
  rm -rf "$APP"
fi
clone "$ELECTRON_APP" "$APP"
rm -f "$RES/default_app.asar"

step "3/7 App files"
mkdir -p "$RES/app/node_modules/vue/dist"
cp "$ROOT/app/main.js" "$ROOT/app/preload.js" "$ROOT/app/package.json" "$RES/app/"
cp -R "$ROOT/app/renderer" "$ROOT/app/build" "$RES/app/"
cp "$ROOT/app/node_modules/vue/dist/vue.global.prod.js" "$RES/app/node_modules/vue/dist/"
mkdir -p "$RES/backend"
rsync -a --exclude '.venv' --exclude '.venv-mlx' --exclude 'tests' --exclude '__pycache__' --exclude '*.pyc' --exclude '*.bak' --exclude 'bench_*.py' \
  "$ROOT/backend/" "$RES/backend/"

step "4/7 Python and its packages"
mkdir -p "$RES/python"
tar -xzf "$TAR" -C "$RES/python" --strip-components 1
SP="$RES/python/lib/python3.12/site-packages"
mkdir -p "$SP"
echo "Adding the packages from backend/.venv (torch, diffusers, …)"
# clone each top-level entry into the new site-packages (keeps pip's own copy from the portable Python)
for item in "$VENV_SP"/* "$VENV_SP"/.[!.]*; do
  [ -e "$item" ] || continue
  name="$(basename "$item")"
  case "$name" in pip|pip-*|_distutils_hack|distutils-precedence.pth) continue ;; esac
  rm -rf "$SP/$name"
  clone "$item" "$SP/$name"
done
PYBIN="$RES/python/bin/python3"
echo "Checking that everything loads…"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$PYBIN" - <<'EOF' || fail "the packaged Python couldn't load its packages (see above)."
import importlib, sys
mods = ["torch", "diffusers", "transformers", "accelerate", "peft", "safetensors", "huggingface_hub", "PIL",
        "cv2", "numpy", "onnxruntime", "facexlib", "refiners", "certifi"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {e}")
import torch
print("Python", sys.version.split()[0], "· torch", torch.__version__, "· Apple GPU:", torch.backends.mps.is_available())
if bad:
    print("Missing:", *bad, sep="\n  ")
    sys.exit(1)
EOF
# Compiled .pyc files record the folder they were compiled in. The ones copied from backend/.venv
# name this Mac's home folder, so they're thrown away and rebuilt with neutral paths.
echo "Compiling Python files with neutral paths (a minute or two)…"
find "$RES/backend" "$SP" -type d -name __pycache__ -prune -exec rm -rf {} +
# (PYTHONDONTWRITEBYTECODE: while compiling, Python itself must not cache its own modules (encodings,
# multiprocessing…) with this Mac's path in them)
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "$PYBIN" -m compileall -q -j 0 -s "$RES" \
  -p "/Applications/$APP_NAME.app/Contents/Resources" "$RES/backend" "$SP" >/dev/null || true
# Any cache file Python still wrote with this Mac's path in it is removed (Python works without it).
# ("|| true": finding none is the normal case, and must not stop the build)
{ grep -rlF "$HOME" "$RES/python" --include='*.pyc' 2>/dev/null || true; } | while IFS= read -r f; do rm -f "$f"; done

# Nothing in the app may mention this Mac's home folder (your username)
LEAKS="$(grep -rlF "$HOME" "$RES" --include='*.pyc' --include='*.py' --include='*.pth' --include='*.json' \
         --include='RECORD' --include='*.cfg' --include='*.txt' --include='*.js' --include='*.html' 2>/dev/null | head -10 || true)"
# ("|| true": finding nothing is the good outcome, and must not stop the build)
if [ -n "$LEAKS" ]; then
  echo "$LEAKS"
  fail "these files in the app mention your home folder (username), see above. Send this text to Claude."
fi
echo "No personal paths found in the app."

# The page template is compiled now, so the app runs Vue without eval under a strict Content Security Policy
cp "$ROOT/app/node_modules/vue/dist/vue.runtime.global.prod.js" "$RES/app/node_modules/vue/dist/"
node "$ROOT/packaging/precompile_template.js" "$RES/app/renderer" || fail "couldn't precompile the page template (see above)."

step "5/7 Name, version and icon"
PL="$APP/Contents/Info.plist"
PB=/usr/libexec/PlistBuddy
plset() { "$PB" -c "Set :$1 $3" "$PL" 2>/dev/null || "$PB" -c "Add :$1 $2 $3" "$PL"; }
plset CFBundleName string "\"$APP_NAME\""
plset CFBundleDisplayName string "\"$APP_NAME\""
plset CFBundleIdentifier string "$BUNDLE_ID"
plset CFBundleShortVersionString string "$VERSION"
plset CFBundleVersion string "$VERSION"
plset CFBundleIconFile string "icon.icns"
plset NSHumanReadableCopyright string "\"Protoart Studio · runs entirely on this Mac\""
plset LSApplicationCategoryType string "public.app-category.graphics-design"
if [ -f "$ROOT/app/build/icon.png" ]; then
  ICONSET="$(mktemp -d)/icon.iconset"
  mkdir -p "$ICONSET"
  for s in 16 32 128 256 512; do
    sips -z $s $s "$ROOT/app/build/icon.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    sips -z $((s * 2)) $((s * 2)) "$ROOT/app/build/icon.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns -o "$RES/icon.icns" "$ICONSET" && rm -f "$RES/electron.icns"
fi

step "6/7 Fast ERNIE engine"
if [ -f "$ROOT/backend/.venv-mlx/.protoart-installed" ] && [ ! -e "$DATA/venv-mlx" ]; then
  echo "Reusing the fast engine you already installed (no download)."
  mkdir -p "$DATA"
  clone "$ROOT/backend/.venv-mlx" "$DATA/venv-mlx"
else
  echo "Nothing to do (it installs itself the first time you use it)."
fi

step "7/7 Hardening and signing"
# Electron fuses: switch off ways to run the app's own engine as a plain Node.js or debug it from outside
if [ -d "$ROOT/app/node_modules/@electron/fuses" ]; then
  (cd "$ROOT/app" && npx --no-install @electron/fuses write --app "$APP" RunAsNode=off \
     EnableNodeOptionsEnvironmentVariable=off EnableNodeCliInspectArguments=off EnableCookieEncryption=on >/dev/null) \
    && echo "Electron fuses set." || echo "Couldn't set Electron fuses (the app still works)."
else
  echo "Skipping Electron fuses: run ./start.sh once to install the build tools."
fi
xattr -cr "$APP" 2>/dev/null || true
if codesign --force --deep --sign - "$APP" >/tmp/protoart_codesign.log 2>&1; then
  echo "Signed (ad hoc, for this Mac)."
else
  echo "Couldn't sign it (details in /tmp/protoart_codesign.log). The app still opens on this Mac."
fi
touch "$APP"

SIZE="$(du -sh "$APP" 2>/dev/null | cut -f1)"
printf '\n\033[32mBuilt: %s (%s)\033[0m\n' "$APP" "$SIZE"
echo "Your images, LoRAs, settings and downloaded models stay where they are (~/.protoartstudio and"
echo "~/.cache/huggingface), so the app and start.sh share them."

if [ "${1:-}" != "--no-install" ]; then
  echo
  read -r -p "Move it to your Applications folder now? [Y/n] " ans || ans="n"
  if [ -z "$ans" ] || [[ "$ans" =~ ^[Yy] ]]; then
    DEST="/Applications/$APP_NAME.app"
    if [ -e "$DEST" ]; then
      echo "Replacing the older copy in Applications…"
      rm -rf "$DEST" || fail "couldn't replace $DEST (is the app still open?). Quit it and run this again."
    fi
    mv "$APP" "$DEST" || fail "couldn't move it to Applications. You can drag it there from the dist folder."
    echo "Done: $DEST"
    read -r -p "Open Protoart Studio now? [Y/n] " o || o="n"
    if [ -z "$o" ] || [[ "$o" =~ ^[Yy] ]]; then open "$DEST"; fi
  else
    open -R "$APP"
  fi
fi
