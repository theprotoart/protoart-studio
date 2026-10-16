#!/bin/bash
# Double-click to make a .dmg installer of Protoart Studio to share (takes about 5-10 minutes).
cd "$(dirname "$0")"
bash packaging/make_dmg.sh "$@"
status=$?
echo
read -r -p "Press Return to close this window." _
exit $status
