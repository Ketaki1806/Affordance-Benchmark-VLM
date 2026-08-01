#!/usr/bin/env bash
# Evaluate FT CLIP + FT Open-VLJEPA on human-filtered pilot captions (N=20).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"

export PYTHONPATH="${PROJECT_ROOT}/src"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${HF_HOME}" \
  "${PROJECT_ROOT}/artifacts/eval/pilot_human"

CONFIG="${1:-${PROJECT_ROOT}/configs/config_eval_human.yaml}"
HUMAN_CAPS="${PROJECT_ROOT}/humaneval/26jul/human_filtered.json"

if [[ ! -f "${HUMAN_CAPS}" ]]; then
  echo "ERROR: missing ${HUMAN_CAPS}"
  echo "Copy humaneval/26jul/human_filtered.json onto the cluster first."
  exit 1
fi

python "${PROJECT_ROOT}/scripts/verify_gpu.py"

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/evaluate-human-$(date +%Y%m%d-%H%M%S).log"
echo "Config: ${CONFIG}"
echo "Captions: ${HUMAN_CAPS}"
echo "Log: ${LOG_FILE}"

python "${PROJECT_ROOT}/src/evaluate.py" --config "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
