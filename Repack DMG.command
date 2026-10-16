#!/bin/bash
# Double-click to re-pack the installer from the app already built in dist/ (about 3 minutes, no rebuild).
cd "$(dirname "$0")"
bash packaging/make_dmg.sh --pack-only
status=$?
echo
read -r -p "Press Return to close this window." _
exit $status
