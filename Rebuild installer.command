#!/bin/bash
# Double-click after an update to make a fresh installer (.dmg) with the latest changes.
# 1) checks the app with its self-tests (about a minute), 2) builds the app and the installer (10-15 minutes),
# 3) shows the new installer and its checksum in Finder. Everything is also written to dmg.log.
cd "$(dirname "$0")"
ROOT="$(pwd)"
VERSION="$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' app/package.json | head -1)"
DMG="dist/Protoart-Studio-$VERSION.dmg"

finish() {
  echo
  read -r -p "Press Return to close this window." _
  exit "$1"
}

echo "== Protoart Studio $VERSION: rebuilding the installer"
echo "   Tip: quit Protoart Studio first, and don't run big jobs while this works."
echo

PY="backend/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "Couldn't find the app's Python (backend/.venv). Start Protoart Studio once from this folder, then try again."
  finish 1
fi

echo "== 1/2 Checking the app (self-tests)…"
"$PY" packaging/run_tests.py 2>&1 | tee tests.log | grep -E "^(FAIL|[0-9]+ tests passed)" || true
if grep -q "^FAIL" tests.log; then
  echo
  echo "Some self-tests failed (details in tests.log). Send tests.log to Claude if you're not sure."
  read -r -p "Build the installer anyway? Type y and press Return, or just Return to stop: " yn
  [ "$yn" = "y" ] || [ "$yn" = "Y" ] || finish 1
fi

echo
echo "== 2/2 Building the app and the installer (10-15 minutes)…"
bash packaging/make_dmg.sh 2>&1 | tee dmg.log
status=${PIPESTATUS[0]}
echo "EXIT $status" >> dmg.log
if [ "$status" -ne 0 ] || [ ! -f "$DMG" ]; then
  echo
  echo "The build stopped with an error (details in dmg.log). Send dmg.log to Claude."
  afplay /System/Library/Sounds/Basso.aiff 2>/dev/null
  finish 1
fi

echo
echo "✅ Done: $DMG"
echo "   Checksum (SHA-256): $(cut -d' ' -f1 "$DMG.sha256")"
echo "   To share it: drag $(basename "$DMG") and $(basename "$DMG").sha256 onto the GitHub draft and Hugging Face."
afplay /System/Library/Sounds/Glass.aiff 2>/dev/null
# (make_dmg.sh already showed it in Finder)
finish 0
