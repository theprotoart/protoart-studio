#!/bin/bash
# Checks Merge LoRAs and embedding training on this Mac with the app's own Python (quit Protoart Studio first).
cd "$(dirname "$0")"
backend/.venv/bin/python backend/tools/selftest_train.py
echo
read -r -p "Press Return to close this window." _
