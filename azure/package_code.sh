#!/usr/bin/env bash
# Build azure/code/ — the FLAT folder of scripts uploaded as the endpoint's code.
#
# score.py imports `los_pred.predict`, which imports `los_pred.features`. Rather
# than keep a second copy of those modules in git, copy them from src/ at package
# time so the endpoint always runs the exact code the project tests.
#
# The layout is flat (all files side by side) because the Azure ML studio
# uploader takes individual files; score.py registers a `los_pred` package over
# its own folder so the imports still resolve (see the shim at the top of it).
#
# Usage (from the repo root):  bash azure/package_code.sh
set -euo pipefail

cd "$(dirname "$0")/.."

OUT=azure/code
rm -rf "$OUT"
mkdir -p "$OUT"

cp azure/score.py "$OUT/score.py"
cp src/los_pred/features.py "$OUT/features.py"
cp src/los_pred/predict.py "$OUT/predict.py"

echo "built $OUT:"
find "$OUT" -type f | sort
