#!/bin/bash
# Double-click to build the standalone "Protoart Studio.app" (takes a few minutes).
cd "$(dirname "$0")"
bash packaging/build_mac_app.sh "$@"
status=$?
echo
read -r -p "Press Return to close this window." _
exit $status
