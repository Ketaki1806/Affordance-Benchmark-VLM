#!/usr/bin/env bash
# Run word–region grounding heatmaps (CLIP + SigLIP).
# Usage:
#   bash scripts/run_grounding.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${PROJECT_ROOT}/artifacts/grounding" "${HF_HOME}"

PAIRS_JSON="${PAIRS_JSON:-${PROJECT_ROOT}/artifacts/eval/val_full/clip.json}"
if [[ ! -f "${PAIRS_JSON}" ]]; then
  PAIRS_JSON="${PROJECT_ROOT}/humaneval/30jul/clip.json"
fi
EXTRA_PAIRS_JSON="${EXTRA_PAIRS_JSON:-${PROJECT_ROOT}/humaneval/1aug/pilot_human/clip_ft.json}"
if [[ ! -f "${EXTRA_PAIRS_JSON}" ]]; then
  EXTRA_PAIRS_JSON="${PROJECT_ROOT}/humaneval/26jul/clip_human.json"
fi

python "${PROJECT_ROOT}/scripts/verify_gpu.py" || true

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/grounding-$(date +%Y%m%d-%H%M%S).log"
echo "Running word grounding. Log: ${LOG_FILE}"
echo "Pairs JSON: ${PAIRS_JSON}"
echo "Extra pairs JSON: ${EXTRA_PAIRS_JSON}"

python "${PROJECT_ROOT}/src/word_grounding.py" \
  --config "${PROJECT_ROOT}/configs/config.yaml" \
  --pairs-json "${PAIRS_JSON}" \
  --extra-pairs-json "${EXTRA_PAIRS_JSON}" \
  --skip-missing \
  --out-dir "${PROJECT_ROOT}/artifacts/grounding" \
  --backends clip siglip \
  --max-words 6 \
  2>&1 | tee "${LOG_FILE}"
