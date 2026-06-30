#!/usr/bin/env bash
# Create environment using micromamba (conda-forge) + pip GPU packages.
# Avoids pytorch-cuda solver issues on HPC clusters.
# Usage: bash scripts/setup_env_micromamba.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="affordance_benchmark"
MAMBA_BIN="${HOME}/bin/micromamba"
MAMBA_ROOT="${HOME}/micromamba"

cd "${PROJECT_ROOT}"

if [[ ! -x "${MAMBA_BIN}" ]]; then
  echo "Micromamba not found. Run: bash scripts/install_micromamba.sh"
  exit 1
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
eval "$("${MAMBA_BIN}" shell hook -s bash -r "${MAMBA_ROOT}")"

if micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Removing broken/partial environment: ${ENV_NAME}"
  micromamba env remove -n "${ENV_NAME}" -y
fi

echo "Creating base environment from environment.yml"
micromamba create -y -n "${ENV_NAME}" -f environment.yml
micromamba activate "${ENV_NAME}"

echo "Installing GPU PyTorch and dependencies via pip"
pip install -r "${PROJECT_ROOT}/requirements-gpu.txt"

echo "Environment ready."
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

echo ""
echo "Next steps (each new SSH session):"
echo "  export MAMBA_ROOT_PREFIX=~/micromamba"
echo "  eval \"\$(~/bin/micromamba shell hook -s bash -r ~/micromamba)\""
echo "  micromamba activate ${ENV_NAME}"
echo "  sbatch scripts/submit_gpu_job.slurm"
