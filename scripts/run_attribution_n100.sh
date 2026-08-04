#!/usr/bin/env bash
# Full N=100 occlusion attribution (modality sensitivity scale-up).
# Skips PNG overlays for speed; writes artifacts/attribution_n100/.
#
# Usage:
#   bash scripts/run_attribution_n100.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"

export PYTHONPATH="${PROJECT_ROOT}/src"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${PROJECT_ROOT}/artifacts/attribution_n100" "${HF_HOME}"

echo "Project root: ${PROJECT_ROOT}"
echo "PYTHONPATH:   ${PYTHONPATH}"
echo ""

PAIRS_JSON="${PAIRS_JSON:-${PROJECT_ROOT}/artifacts/eval/val_full/clip.json}"
if [[ ! -f "${PAIRS_JSON}" ]]; then
  PAIRS_JSON="${PROJECT_ROOT}/humaneval/30jul/clip.json"
fi

python "${PROJECT_ROOT}/scripts/verify_gpu.py" || true

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/attribution_n100-$(date +%Y%m%d-%H%M%S).log"
echo "Running N=100 occlusion attribution. Log: ${LOG_FILE}"
echo "Pairs JSON: ${PAIRS_JSON}"

python "${PROJECT_ROOT}/src/attribution_occlusion.py" \
  --config "${PROJECT_ROOT}/configs/config.yaml" \
  --pairs-json "${PAIRS_JSON}" \
  --all-pairs \
  --no-overlays \
  --out-dir "${PROJECT_ROOT}/artifacts/attribution_n100" \
  --backends clip siglip open_vljepa \
  --vljepa-checkpoint "${PROJECT_ROOT}/artifacts/checkpoints/open-vljepa/best.pt" \
  2>&1 | tee "${LOG_FILE}"
