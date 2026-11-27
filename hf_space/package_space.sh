#!/usr/bin/env bash
# Assemble the Hugging Face Space (Docker SDK) into build/hf-space/.
#
# A Space is its own git repo, so it gets only what the slim serving image
# needs, laid out as Hugging Face expects:
#
#   build/hf-space/
#     README.md        <- Space card (YAML header: sdk: docker, app_port: 8000)
#     Dockerfile       <- copy of ../Dockerfile.slim
#     pyproject.toml   <- for the `serving` dependency group
#     uv.lock
#     src/los_pred/    <- app code
#     serving_model/   <- the baked @champion bundle (gitignored in this repo)
#
# Prereq:  uv run python -m los_pred.export_model
# Run:     bash hf_space/package_space.sh
# Then upload build/hf-space/ to the Space (see the README's deployment section).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build/hf-space"

if [ ! -f "$ROOT/serving_model/metadata.json" ]; then
  echo "serving_model/ not found. Run: uv run python -m los_pred.export_model" >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT/src"

cp "$ROOT/hf_space/README.md" "$OUT/README.md"
cp "$ROOT/Dockerfile.slim"    "$OUT/Dockerfile"
cp "$ROOT/uv.lock" "$OUT/"
rsync -a --exclude '__pycache__' --exclude '.DS_Store' "$ROOT/src/los_pred" "$OUT/src/"
rsync -a --exclude '.DS_Store' "$ROOT/serving_model" "$OUT/"

# --- scrub personal details (the Space can be public) ---------------------
# pyproject.toml: drop the `authors = [...]` block (name + email). The image
# never installs the project itself, and uv.lock doesn't record authors, so
# `uv sync --frozen` is unaffected.
sed -e '/^authors = \[/,/^\]/d' "$ROOT/pyproject.toml" > "$OUT/pyproject.toml"
# MLmodel: `artifact_path` is the absolute path of the training machine's
# mlruns/ folder (it shows the local username). MLflow doesn't use it when
# loading a model from a local folder, so replace it with a neutral value.
sed -e 's|^artifact_path: .*|artifact_path: artifacts|' \
  "$ROOT/serving_model/model/MLmodel" > "$OUT/serving_model/model/MLmodel"

# Fail if anything that looks like a local home path or an email address remains.
if grep -rnE '/Users/[A-Za-z]|/home/[A-Za-z]|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}' \
     "$OUT" --exclude=uv.lock --exclude=export_model.py; then
  echo "Personal details found above; not safe to upload." >&2
  exit 1
fi

echo "Space assembled at $OUT (model: $(cat "$OUT/serving_model/metadata.json" | tr -d '\n '))"
find "$OUT" -type f | sed "s|$OUT/|  |" | sort
