#!/usr/bin/env bash
# SigLIP-only: N=100 occlusion attribution + embedding modality gap.
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

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/siglip_n100-$(date +%Y%m%d-%H%M%S).log"
echo "SigLIP N=100. Log: ${LOG_FILE}"
echo "Pairs JSON: ${PAIRS_JSON}"
python "${PROJECT_ROOT}/scripts/verify_gpu.py" || true

{
  echo "=== occlusion attribution (siglip) ==="
  python "${PROJECT_ROOT}/src/attribution_occlusion.py" \
    --config "${PROJECT_ROOT}/configs/config.yaml" \
    --pairs-json "${PAIRS_JSON}" \
    --all-pairs \
    --no-overlays \
    --backends siglip \
    --out-dir "${PROJECT_ROOT}/artifacts/attribution_n100"

  echo "=== embedding modality gap (siglip) ==="
  python "${PROJECT_ROOT}/src/compute_modality_gap.py" \
    --config "${PROJECT_ROOT}/configs/config.yaml" \
    --pairs-json "${PAIRS_JSON}" \
    --all-pairs \
    --backends siglip \
    --out-dir "${PROJECT_ROOT}/artifacts/attribution_n100"
} 2>&1 | tee "${LOG_FILE}"
