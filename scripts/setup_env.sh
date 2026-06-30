#!/usr/bin/env bash
# Create or update the conda environment from environment.yml + pip GPU packages.
# Usage (from project root):
#   bash scripts/setup_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="affordance_benchmark"

cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/load_conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Updating existing environment: ${ENV_NAME}"
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  echo "Creating environment: ${ENV_NAME}"
  conda env create -f environment.yml
fi

conda activate "${ENV_NAME}"

echo "Installing GPU PyTorch and dependencies via pip"
pip install -r "${PROJECT_ROOT}/requirements-gpu.txt"

echo "Environment ready."
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

echo ""
echo "Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  bash scripts/run_pipeline.sh"
