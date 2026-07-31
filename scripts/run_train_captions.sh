#!/usr/bin/env bash
# Caption the train_500 manifest with Qwen (uses configs/config_train_ft.yaml).
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

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/pipeline-train500-$(date +%Y%m%d-%H%M%S).log"
echo "Running train caption pipeline. Log: ${LOG_FILE}"

python "${PROJECT_ROOT}/src/pipeline.py" \
  --config "${PROJECT_ROOT}/configs/config_train_ft.yaml" \
  "$@" 2>&1 | tee "${LOG_FILE}"
