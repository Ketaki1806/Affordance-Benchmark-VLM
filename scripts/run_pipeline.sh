#!/usr/bin/env bash
# Run the caption generation pipeline on GPU.
# Usage (from project root, with conda env active):
#   bash scripts/run_pipeline.sh
#   bash scripts/run_pipeline.sh --shard-index 0 --num-shards 10

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
echo "HF_HOME:      ${HF_HOME}"
echo "Args:         $*"
echo ""

python "${PROJECT_ROOT}/scripts/verify_gpu.py"

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/pipeline-$(date +%Y%m%d-%H%M%S).log"
echo "Running pipeline. Log: ${LOG_FILE}"

python "${PROJECT_ROOT}/src/pipeline.py" "$@" 2>&1 | tee "${LOG_FILE}"
