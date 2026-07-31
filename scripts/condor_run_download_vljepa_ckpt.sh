#!/usr/bin/env bash
# HTCondor wrapper: download Open-VLJEPA best.pt on a worker (not login interactive).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p artifacts/logs

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"

echo "Job ID:     ${CONDOR_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Started at: $(date)"

bash "${PROJECT_ROOT}/scripts/download_open_vljepa_ckpt.sh"

echo "Finished at: $(date)"
