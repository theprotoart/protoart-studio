#!/bin/bash
# Times Enhance & upscale (Turbo / Fast / Standard) on this Mac. Results: enhance_bench.log + enhance_bench/
cd "$(dirname "$0")"
echo "Enhance speed test (a few minutes; close Protoart Studio first so it has the memory)…"
PYTHONNOUSERSITE=1 backend/.venv/bin/python backend/bench_enhance.py 2>&1 | tee enhance_bench.log
echo
read -r -p "Done. Press Return to close this window." _
