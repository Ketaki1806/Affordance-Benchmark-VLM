#!/usr/bin/env bash
# Run the caption generation pipeline on GPU.
# Usage (from project root, with conda env active):
#   bash scripts/run_pipeline.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="affordance_benchmark"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

export PYTHONPATH="${PROJECT_ROOT}/src"
export HF_HOME="${PROJECT_ROOT}/artifacts/huggingface_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${HF_HOME}"

echo "Project root: ${PROJECT_ROOT}"
echo "PYTHONPATH:   ${PYTHONPATH}"
echo "HF_HOME:      ${HF_HOME}"
echo ""

python "${PROJECT_ROOT}/scripts/verify_gpu.py"

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/pipeline-$(date +%Y%m%d-%H%M%S).log"
echo "Running pipeline. Log: ${LOG_FILE}"

python "${PROJECT_ROOT}/src/pipeline.py" 2>&1 | tee "${LOG_FILE}"
