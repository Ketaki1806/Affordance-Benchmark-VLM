#!/usr/bin/env bash
# HTCondor: SigLIP + Y-space on N=100.
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

bash "${PROJECT_ROOT}/scripts/run_siglip_yspace.sh"

echo "Finished at: $(date)"
