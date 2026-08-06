#!/usr/bin/env bash
# Fine-tune CLIP on train_1000 filtered captions.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${HF_HOME}"

CFG="${CFG:-${PROJECT_ROOT}/configs/config_train_1000.yaml}"
FILTERED="${PROJECT_ROOT}/artifacts/captions/train_1000/filtered.json"
if [[ ! -f "${FILTERED}" ]]; then
  echo "ERROR: missing ${FILTERED}" >&2
  echo "Caption + merge first." >&2
  exit 1
fi

python "${PROJECT_ROOT}/scripts/verify_gpu.py" || true
LOG_FILE="${PROJECT_ROOT}/artifacts/logs/finetune-clip-train1000-$(date +%Y%m%d-%H%M%S).log"
echo "CLIP FT train_1000. Log: ${LOG_FILE}"

python "${PROJECT_ROOT}/src/finetune_clip.py" \
  --config "${CFG}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
