#!/usr/bin/env bash
# Build PACO-LVIS train manifest with N=1000 (one preferred part per image).
#
# Usage (on cluster, after COCO train2017 + PACO train ann are available):
#   source scripts/activate_env.sh
#   bash scripts/build_paco_train_1000.sh
#
# Optional overrides:
#   IMAGE_ROOT=data/paco/coco N=1000 SEED=42 bash scripts/build_paco_train_1000.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v python >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/activate_env.sh"
fi

export PYTHONPATH="${PROJECT_ROOT}/src"

ANN="${ANN:-${PROJECT_ROOT}/data/paco/annotations/paco_lvis_v1_train.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${PROJECT_ROOT}/data/paco/coco}"
OUT="${OUT:-${PROJECT_ROOT}/data/paco/manifest_train_1000.json}"
N="${N:-1000}"
SEED="${SEED:-42}"
EXCLUDE="${EXCLUDE:-${PROJECT_ROOT}/data/paco/manifest_val_100.json}"

if [[ ! -f "${ANN}" ]]; then
  echo "ERROR: missing train annotations: ${ANN}" >&2
  echo "Download PACO ann zip and unzip paco_lvis_v1_train.json under data/paco/annotations/" >&2
  exit 1
fi

ARGS=(
  --ann "${ANN}"
  --image-root "${IMAGE_ROOT}"
  --output "${OUT}"
  --n "${N}"
  --seed "${SEED}"
  --source-split train
  --require-image
)

if [[ -f "${EXCLUDE}" ]]; then
  ARGS+=(--exclude-manifest "${EXCLUDE}")
  echo "Holding out eval ids from ${EXCLUDE}"
else
  echo "NOTE: ${EXCLUDE} not found — not excluding val_100 ids (PACO train/val should still be disjoint)."
fi

echo "Building train_${N} manifest -> ${OUT}"
python "${PROJECT_ROOT}/src/build_paco_val_manifest.py" "${ARGS[@]}"
echo "Done. Next: bash scripts/condor_submit_train_1000_captions_sharded.sh"
