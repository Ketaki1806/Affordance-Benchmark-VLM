#!/usr/bin/env bash
# HTCondor: fine-tune Open-VLJEPA on train_500 captions.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p artifacts/logs

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"

echo "Job ID:     ${CONDOR_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Started at: $(date)"
nvidia-smi || true

bash "${PROJECT_ROOT}/scripts/run_finetune_open_vljepa.sh"

echo "Finished at: $(date)"
