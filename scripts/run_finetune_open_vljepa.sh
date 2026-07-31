#!/usr/bin/env bash
# Fine-tune Open-VLJEPA on train_500 affordance pairs.
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

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/finetune-vljepa-$(date +%Y%m%d-%H%M%S).log"
echo "Running Open-VLJEPA fine-tune. Log: ${LOG_FILE}"

python "${PROJECT_ROOT}/src/finetune_open_vljepa.py" \
  --config "${PROJECT_ROOT}/configs/config_train_ft.yaml" \
  "$@" 2>&1 | tee "${LOG_FILE}"
