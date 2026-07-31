#!/usr/bin/env bash
# HTCondor wrapper: upgrade torch on a worker (avoids login SSH drops / OOM).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p artifacts/logs

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"

echo "Job ID:     ${CONDOR_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Started at: $(date)"

bash "${PROJECT_ROOT}/scripts/upgrade_torch.sh"

echo "Finished at: $(date)"
