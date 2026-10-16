#!/bin/bash
# Makes a .dmg installer of Protoart Studio to share with friends.
#   1. builds a fresh, properly signed app (packaging/build_mac_app.sh)
#   2. checks that nothing in it depends on programs only installed on this Mac
#   3. packs it with an Applications shortcut and a "Read me first" note
# Result: dist/Protoart-Studio-<version>.dmg (about 0.8-1 GB)
#
# Run it with "Make DMG for friends.command" (double-click) or:  bash packaging/make_dmg.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
APP_NAME="Protoart Studio"
APP="$ROOT/dist/$APP_NAME.app"
VERSION="$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' "$ROOT/app/package.json" | head -1)"
DMG="$ROOT/dist/Protoart-Studio-${VERSION}.dmg"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mStopped: %s\033[0m\n' "$*"; exit 1; }

if [ "${1:-}" = "--pack-only" ] && [ -d "$APP" ]; then
  step "Using the app already built in dist/ (--pack-only)"
else
  step "Building a fresh copy of the app"
  bash "$ROOT/packaging/build_mac_app.sh" --no-install
fi
[ -d "$APP" ] || fail "the build didn't produce $APP."

step "Checking it will run on other Macs"
SP="$APP/Contents/Resources/python/lib/python3.12/site-packages"
PROBLEMS="$(mktemp)"
# compiled libraries that link to Homebrew (or other local installs) won't load on a friend's Mac
find "$APP/Contents/Resources/python" \( -name '*.so' -o -name '*.dylib' \) -type f -print0 |
  while IFS= read -r -d '' f; do
    # (a library's own name, its "install name", can be a leftover path from the machine that built the
    # package, e.g. /Users/runner/... on a build server: that's a label, not something it needs, so skip it)
    own="$(otool -D "$f" 2>/dev/null | tail -n +2 | head -1)"
    # (universal libraries list one block per chip type: each block's header line, ending in ':', is the
    # file's own location in dist/ on this Mac, not a dependency, so header lines are skipped too)
    if otool -L "$f" 2>/dev/null | { grep -v ':$' || true; } | awk '{print $1}' | { grep -vxF "${own:-/nonexistent}" || true; } |
         grep -qE '^(/opt/homebrew|/usr/local/(opt|Cellar|lib)|/Users/)'; then
      echo "  ${f#$SP/}" >> "$PROBLEMS"
    fi
  done
# packages installed "in place" from a folder on this Mac point back to that folder
grep -lE '^/Users/' "$SP"/*.pth 2>/dev/null | sed 's|^|  |' >> "$PROBLEMS" || true
if [ -s "$PROBLEMS" ]; then
  echo "These parts depend on files that only exist on this Mac, so they may not work for friends:"
  head -20 "$PROBLEMS"
  echo
  read -r -p "Make the DMG anyway? [y/N] " a || a="n"
  [[ "$a" =~ ^[Yy] ]] || fail "stopped so these can be fixed first (send this text to Claude)."
else
  echo "Looks self-contained: no links to Homebrew or folders on this Mac."
fi
rm -f "$PROBLEMS"
codesign --verify --deep "$APP" >/dev/null 2>&1 && echo "Signature OK." || echo "Note: the signature couldn't be verified (the app may show a 'damaged' message on other Macs)."

step "Packing the DMG"
# Leftovers from earlier runs that stopped half-way (each can hold a ~2 GB disk image): remove them first.
TMPROOT="${TMPDIR:-/tmp}"
for d in "$TMPROOT"/tmp.*; do
  [ -d "$d/Protoart Studio" ] || [ -e "$d/rw.dmg" ] || [ -e "$d/rw.sparseimage" ] || continue
  rm -rf "$d"
done
for v in "/Volumes/$APP_NAME" "/Volumes/$APP_NAME "*; do [ -d "$v" ] && hdiutil detach "$v" -force >/dev/null 2>&1 || true; done
WORK="$(mktemp -d)"
MOUNT=""
cleanup() {
  [ -n "$MOUNT" ] && [ -d "$MOUNT" ] && { hdiutil detach "$MOUNT" >/dev/null 2>&1 || hdiutil detach "$MOUNT" -force >/dev/null 2>&1; }
  rm -rf "$WORK"
}
trap cleanup EXIT  # always remove the temporary copy and disk image, also when something fails
STAGE="$WORK/Protoart Studio"
mkdir -p "$STAGE"
cp -cR "$APP" "$STAGE/" 2>/dev/null || cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cat > "$STAGE/Read me first.txt" <<EOF
Protoart Studio ${VERSION}: test version
========================================

CHECK THE DOWNLOAD (recommended)
Each release lists a SHA-256 checksum. In Terminal, run
    shasum -a 256 ~/Downloads/Protoart-Studio-${VERSION}.dmg
and compare the long code it prints with the one on the release page. They must match exactly.

INSTALL
1. Drag "Protoart Studio" onto the Applications folder in this window.
2. Open Applications and double-click Protoart Studio.

THE FIRST TIME, macOS WILL WARN YOU
This test version isn't signed with an Apple developer certificate yet, so macOS can't check it.
  * macOS 15 (Sequoia) or newer: click Done, then open System Settings > Privacy & Security,
    scroll down and click "Open Anyway" next to Protoart Studio, then confirm.
  * Older macOS: right-click (or Control-click) the app, choose Open, then Open again.
You only need to do this once.

WHAT YOU NEED
  * A Mac with Apple Silicon (M1 or newer). 16 GB of memory is comfortable; 8 GB works with
    SD 1.5 and smaller sizes.
  * Free disk space for models: each one downloads the first time you use it (SD 1.5 ~2 GB,
    SDXL ~5-7 GB, ERNIE ~7 GB), then works offline.

PRIVATE BY DESIGN
Everything runs on your Mac. Prompts and images never leave it; no account, no tracking. It only
goes online to download a model or tool you ask for that isn't on your Mac yet.
Your images and settings are kept in the hidden folder ~/.protoartstudio.

COMMON PROBLEMS
  * "is damaged and can't be opened": the download was changed or broken on the way. Don't try to
    force it open. Delete it, download it again from the official release page, and check the
    checksum (see CHECK THE DOWNLOAD above).
  * Downloads fail ("couldn't connect", "Bad file descriptor"): a firewall app such as Little Snitch
    or LuLu may be blocking it. The app falls back to macOS's own downloader automatically; if it
    still fails, allow "python3.12" (Protoart Studio) in your firewall app.
  * "The Python backend stopped" or the Mac freezes: the picture needed more memory than the Mac has.
    Click Restart backend and use a smaller size or fewer images at once.
  * Very slow: SD 1.5 at 512 px and Fast mode are quickest on 8 GB Macs; the first picture after
    opening the app is always slower.
More fixes: How to / About -> Troubleshooting in the app.

If you have DiffusionBee, your models, LoRAs and embeddings show up automatically.
See the "How to / About" tab in the app for a guide to every feature.
EOF

rm -f "$DMG"
FREE_GB=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
[ "${FREE_GB:-0}" -ge 3 ] || fail "only ${FREE_GB:-0} GB free; making the DMG needs about 3 GB while it works (empty the Trash, or free some room, then try again)."

# The window friends see when they open the DMG: the app on the left, an arrow, Applications on the right.
BG_OK=0
mkdir -p "$STAGE/.background"
if "$ROOT/backend/.venv/bin/python" "$ROOT/packaging/dmg_background.py" "$STAGE/.background" >/dev/null 2>&1; then
  tiffutil -cathidpicheck "$STAGE/.background/bg.png" "$STAGE/.background/bg@2x.png" -out "$STAGE/.background/bg.tiff" >/dev/null 2>&1 &&
    rm -f "$STAGE/.background/bg.png" "$STAGE/.background/bg@2x.png" && BG_OK=1
fi
echo "Laying out the install window…"
# Size the image from the files' real (uncompressed) size: macOS stores some of them compressed, and du alone
# reports that smaller size, which made the image too small. A sparse image only uses what's written, so
# generous headroom costs nothing.
APPARENT_MB=$(du -sAm "$STAGE" 2>/dev/null | cut -f1)
RW_MB=$(( ${APPARENT_MB:-0} * 3 / 2 + 600 ))
[ "$RW_MB" -ge $(( $(du -sm "$STAGE" | cut -f1) * 2 + 600 )) ] || RW_MB=$(( $(du -sm "$STAGE" | cut -f1) * 2 + 600 ))
# A sparse image only takes the room its files use (no full-size blank image, no hidden second copy)
hdiutil create -volname "$APP_NAME" -size "${RW_MB}m" -fs HFS+ -type SPARSE -ov "$WORK/rw" >/dev/null ||
  fail "hdiutil couldn't create the DMG ($(df -h "$ROOT" | awk 'NR==2 {print $4}') free)."
RW="$WORK/rw.sparseimage"
MOUNT="$(hdiutil attach -readwrite -noverify -noautoopen "$RW" | awk -F'\t' '/\/Volumes\// {print $NF}' | tail -1)"
[ -n "$MOUNT" ] && [ -d "$MOUNT" ] || fail "couldn't open the new disk image."
ditto "$STAGE/" "$MOUNT/" || fail "couldn't copy the app into the disk image ($(df -h "$ROOT" | awk 'NR==2 {print $4}') free)."
rm -rf "$STAGE"  # its copy is in the image now
STYLED=0
if [ -n "$MOUNT" ] && [ -d "$MOUNT" ]; then
  DISK="$(basename "$MOUNT")"
  BGLINE=""
  [ "$BG_OK" = 1 ] && BGLINE='set background picture of opts to file ".background:bg.tiff"'
  # (the first time, macOS asks whether Terminal may control Finder: allow it, so the window is laid out)
  if osascript <<OSA >/dev/null 2>&1
tell application "Finder"
  tell disk "$DISK"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set bounds of container window to {200, 120, 860, 608}
    set opts to the icon view options of container window
    set arrangement of opts to not arranged
    set icon size of opts to 96
    set text size of opts to 13
    $BGLINE
    set position of item "$APP_NAME.app" of container window to {170, 160}
    set position of item "Applications" of container window to {490, 160}
    set position of item "Read me first.txt" of container window to {330, 335}
    update without registering applications
    delay 1
    close
  end tell
end tell
OSA
  then STYLED=1; fi
  [ -d "$MOUNT/.fseventsd" ] && rm -rf "$MOUNT/.fseventsd" || true
  sync
  hdiutil detach "$MOUNT" >/dev/null 2>&1 || hdiutil detach "$MOUNT" -force >/dev/null 2>&1 || true
  MOUNT=""
fi
[ "$STYLED" = 1 ] && echo "Install window ready: app → arrow → Applications." ||
  echo "Note: couldn't lay out the window (Finder automation not allowed?); the DMG still has the Applications shortcut."
echo "Compressing (a few minutes)…"
hdiutil convert "$RW" -format ULFO -o "$DMG" >/dev/null 2>&1 ||
  hdiutil convert "$RW" -format UDZO -imagekey zlib-level=9 -o "$DMG" >/dev/null ||
  fail "hdiutil couldn't compress the DMG."
rm -rf "$WORK"
hdiutil verify "$DMG" >/dev/null 2>&1 && echo "DMG checked OK."
(cd "$(dirname "$DMG")" && shasum -a 256 "$(basename "$DMG")" > "$(basename "$DMG").sha256")
SHA="$(cut -d' ' -f1 "$DMG.sha256")"
echo "SHA-256: $SHA"
echo "(saved next to the DMG as $(basename "$DMG").sha256: publish it with the download so people can check it)"

SIZE="$(du -h "$DMG" | cut -f1)"
printf '\n\033[32mReady to share: %s (%s)\033[0m\n' "$DMG" "$SIZE"
echo "Send it with AirDrop, or upload it (GitHub Releases, iCloud Drive, Google Drive, Dropbox) with its checksum"
echo "(it's too big for email). Friends: open it, drag the app to Applications, and follow 'Read me first'."
open -R "$DMG"
