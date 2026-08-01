#!/usr/bin/env bash
# Train manifest: N=500, exclude manifest_val_100.json
#   bash scripts/build_paco_train_manifest.sh --image-root data/paco/coco --require-image

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v python >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/activate_env.sh"
fi

export PYTHONPATH="${PROJECT_ROOT}/src"

python "${PROJECT_ROOT}/src/build_paco_val_manifest.py" \
  --n 500 \
  --seed 43 \
  --exclude-manifest "${PROJECT_ROOT}/data/paco/manifest_val_100.json" \
  --output "${PROJECT_ROOT}/data/paco/manifest_train_500.json" \
  --source-split train_holdout \
  "$@"
