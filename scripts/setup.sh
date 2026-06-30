#!/usr/bin/env bash
# Create micromamba env and install GPU dependencies (LST / Linux).
# Usage: bash scripts/setup.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="affordance_benchmark"
MAMBA_BIN="${HOME}/bin/micromamba"

cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"

if [[ ! -x "${MAMBA_BIN}" ]]; then
  echo "Micromamba not found. Run: bash scripts/install_micromamba.sh"
  exit 1
fi

eval "$("${MAMBA_BIN}" shell hook -s bash -r "${MAMBA_ROOT_PREFIX}")"

if ! micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Creating environment: ${ENV_NAME}"
  micromamba create -y -n "${ENV_NAME}" -f environment.yml
fi

micromamba activate "${ENV_NAME}"

echo "Installing GPU PyTorch and dependencies via pip..."
pip install --no-cache-dir -r "${PROJECT_ROOT}/requirements-gpu.txt"

echo ""
echo "Environment ready."
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

echo ""
echo "Each new SSH session:"
echo "  source scripts/activate_env.sh"
echo ""
echo "Run on a GPU node:"
echo "  sbatch scripts/submit_gpu_job.slurm"
