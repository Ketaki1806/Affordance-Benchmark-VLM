#!/usr/bin/env bash
# Install GPU pip packages — run on a compute node if login node OOM-kills pip.
# Usage:
#   bash scripts/install_deps.sh          # interactive (compute node or tmux)
#   sbatch scripts/install_deps.slurm     # batch job

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="affordance_benchmark"
MAMBA_BIN="${HOME}/bin/micromamba"
REQ_FILE="${PROJECT_ROOT}/requirements-gpu.txt"

cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"

if [[ ! -x "${MAMBA_BIN}" ]]; then
  echo "ERROR: Run bash scripts/install_micromamba.sh first."
  exit 1
fi

eval "$("${MAMBA_BIN}" shell hook -s bash -r "${MAMBA_ROOT_PREFIX}")"
micromamba activate "${ENV_NAME}"

echo "Installing on: $(hostname)"
echo "Started at:    $(date)"

# Two-step install uses less peak RAM than resolving everything at once.
echo ""
echo "[1/2] Installing PyTorch (cu121)..."
pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.5.1"

echo ""
echo "[2/2] Installing transformers and dependencies..."
pip install --no-cache-dir \
  "transformers>=4.46.0" \
  "accelerate>=0.34.0" \
  "huggingface_hub>=0.26.0" \
  "sentencepiece>=0.2.0" \
  "safetensors>=0.4.0"

echo ""
echo "Done at: $(date)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
