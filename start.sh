#!/bin/bash
cd "$(dirname "$0")"
if [ ! -x backend/.venv/bin/python ] || [ ! -d app/node_modules ]; then
  echo "Run ./setup.sh first."
  exit 1
fi
# After an update that changes the Electron version, install the new one first.
WANT="$(sed -n 's/.*"electron": *"\([0-9.]*\)".*/\1/p' app/package.json | head -1)"
HAVE="$(node -p "require('./app/node_modules/electron/package.json').version" 2>/dev/null)"
if [ -n "$WANT" ] && [ "$WANT" != "$HAVE" ]; then
  echo "Updating Electron ${HAVE:-?} -> $WANT (one time, ~100 MB)…"
  (cd app && npm install) || exit 1
  (cd app && node node_modules/electron/install.js) || exit 1
fi
if [ ! -f app/node_modules/electron/path.txt ]; then
  echo "Finishing Electron install…"
  (cd app && node node_modules/electron/install.js) || exit 1
fi
cd app && npm start
