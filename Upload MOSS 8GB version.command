#!/bin/bash
# Publishes MOSS-Image's 8 GB version (made by the app on this Mac) to Protoart Studio's Hugging Face page.
cd "$(dirname "$0")"
backend/.venv/bin/python backend/tools/upload_moss_small.py
echo
read -r -p "Press Return to close this window." _
