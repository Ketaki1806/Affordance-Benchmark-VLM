#!/usr/bin/env bash
# Run stage 4 evaluation (CLIP + optional Open-VLJEPA) on filtered captions.
# Usage:
#   bash scripts/run_evaluate.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"

export PYTHONPATH="${PROJECT_ROOT}/src"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${HF_HOME}"

echo "Project root: ${PROJECT_ROOT}"
echo "PYTHONPATH:   ${PYTHONPATH}"
echo ""

python "${PROJECT_ROOT}/scripts/verify_gpu.py"

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/evaluate-$(date +%Y%m%d-%H%M%S).log"
echo "Running evaluation. Log: ${LOG_FILE}"

python "${PROJECT_ROOT}/src/evaluate.py" 2>&1 | tee "${LOG_FILE}"
