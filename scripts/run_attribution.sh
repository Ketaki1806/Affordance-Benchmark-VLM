#!/usr/bin/env bash
# Run occlusion attribution (text leave-one-out + grid blackout) on eval pairs.
# Usage:
#   bash scripts/run_attribution.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"

export PYTHONPATH="${PROJECT_ROOT}/src"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${PROJECT_ROOT}/artifacts/attribution" "${HF_HOME}"

echo "Project root: ${PROJECT_ROOT}"
echo "PYTHONPATH:   ${PYTHONPATH}"
echo ""

PAIRS_JSON="${PAIRS_JSON:-${PROJECT_ROOT}/artifacts/eval/val_full/clip.json}"
if [[ ! -f "${PAIRS_JSON}" ]]; then
  PAIRS_JSON="${PROJECT_ROOT}/humaneval/30jul/clip.json"
fi

# The N=100 eval JSONs are missing lvis_258649 (blender); it only exists in the pilot
# human-eval JSONs, so fall back there for whichever image_ids the primary source lacks.
EXTRA_PAIRS_JSON="${EXTRA_PAIRS_JSON:-${PROJECT_ROOT}/humaneval/1aug/pilot_human/clip_ft.json}"
if [[ ! -f "${EXTRA_PAIRS_JSON}" ]]; then
  EXTRA_PAIRS_JSON="${PROJECT_ROOT}/humaneval/26jul/clip_human.json"
fi

python "${PROJECT_ROOT}/scripts/verify_gpu.py" || true

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/attribution-$(date +%Y%m%d-%H%M%S).log"
echo "Running occlusion attribution. Log: ${LOG_FILE}"
echo "Pairs JSON: ${PAIRS_JSON}"
echo "Extra pairs JSON: ${EXTRA_PAIRS_JSON}"

python "${PROJECT_ROOT}/src/attribution_occlusion.py" \
  --config "${PROJECT_ROOT}/configs/config.yaml" \
  --pairs-json "${PAIRS_JSON}" \
  --extra-pairs-json "${EXTRA_PAIRS_JSON}" \
  --skip-missing \
  --out-dir "${PROJECT_ROOT}/artifacts/attribution" \
  --backends clip siglip open_vljepa \
  --vljepa-checkpoint "${PROJECT_ROOT}/artifacts/checkpoints/open-vljepa/best.pt" \
  2>&1 | tee "${LOG_FILE}"
