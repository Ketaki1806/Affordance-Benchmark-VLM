#!/usr/bin/env bash
# SigLIP eval (N=100) then EmbeddingGemma Y-space analysis.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"

export PYTHONPATH="${PROJECT_ROOT}/src"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${HF_HOME}"

python "${PROJECT_ROOT}/scripts/verify_gpu.py"

SIGLIP_CFG="${PROJECT_ROOT}/configs/config_eval_siglip.yaml"
FILTERED="${PROJECT_ROOT}/artifacts/captions/val_full/filtered.json"
CLIP_JSON="${PROJECT_ROOT}/artifacts/eval/val_full/clip.json"
YSPACE_OUT="${PROJECT_ROOT}/artifacts/eval/val_full/yspace_caption_analysis.json"

if [[ ! -f "${FILTERED}" ]]; then
  echo "ERROR: missing ${FILTERED}"
  exit 1
fi

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/siglip-yspace-$(date +%Y%m%d-%H%M%S).log"
echo "Log: ${LOG_FILE}"

{
  echo "=== SigLIP ==="
  python "${PROJECT_ROOT}/src/evaluate.py" --config "${SIGLIP_CFG}"

  echo "=== Y-space (EmbeddingGemma) ==="
  YSPACE_ARGS=(
    --captions "${FILTERED}"
    --out "${YSPACE_OUT}"
  )
  if [[ -f "${CLIP_JSON}" ]]; then
    YSPACE_ARGS+=(--clip-json "${CLIP_JSON}")
  fi
  python "${PROJECT_ROOT}/src/analyze_caption_yspace.py" "${YSPACE_ARGS[@]}"
} 2>&1 | tee "${LOG_FILE}"
