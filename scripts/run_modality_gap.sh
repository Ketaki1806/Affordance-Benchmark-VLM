#!/usr/bin/env bash
# Embedding modality gap on N=100 (alignment geometry; not occlusion vision_share).
#
# Usage:
#   bash scripts/run_modality_gap.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${PROJECT_ROOT}/artifacts/attribution_n100" "${HF_HOME}"

PAIRS_JSON="${PAIRS_JSON:-${PROJECT_ROOT}/artifacts/eval/val_full/clip.json}"
if [[ ! -f "${PAIRS_JSON}" ]]; then
  PAIRS_JSON="${PROJECT_ROOT}/humaneval/30jul/clip.json"
fi

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/modality_gap-$(date +%Y%m%d-%H%M%S).log"
echo "Running embedding modality gap. Log: ${LOG_FILE}"
echo "Pairs JSON: ${PAIRS_JSON}"

python "${PROJECT_ROOT}/scripts/verify_gpu.py" || true

python "${PROJECT_ROOT}/src/compute_modality_gap.py" \
  --config "${PROJECT_ROOT}/configs/config.yaml" \
  --pairs-json "${PAIRS_JSON}" \
  --all-pairs \
  --out-dir "${PROJECT_ROOT}/artifacts/attribution_n100" \
  --backends clip siglip open_vljepa \
  --vljepa-checkpoint "${PROJECT_ROOT}/artifacts/checkpoints/open-vljepa/best.pt" \
  2>&1 | tee "${LOG_FILE}"
