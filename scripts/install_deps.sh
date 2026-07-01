#!/usr/bin/env bash
# Install GPU packages. PyTorch via micromamba (login-node safe); rest via pip.
# Usage:
#   bash scripts/install_deps.sh
# On LST submit node (HTCondor):
#   bash scripts/condor_submit_install.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="affordance_benchmark"
MAMBA_BIN="${HOME}/bin/micromamba"

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
echo ""
echo "NOTE: pip install of PyTorch OOM-kills on login nodes (shows 'Killed')."
echo "      Using micromamba for PyTorch instead."
echo ""

echo "[1/2] Installing PyTorch + CUDA 12.1 via micromamba (may take 5–15 min)..."
micromamba install -y \
  pytorch pytorch-cuda=12.1 \
  -c pytorch -c nvidia

echo ""
echo "[2/2] Installing transformers and dependencies via pip..."
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
