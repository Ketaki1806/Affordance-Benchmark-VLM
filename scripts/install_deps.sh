#!/usr/bin/env bash
# Install GPU packages. PyTorch via micromamba; pip fallback on execution nodes.
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

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"

echo "Installing on: $(hostname)"
echo "Started at:    $(date)"
echo ""

install_pytorch_micromamba() {
  echo "[1/2] Installing PyTorch + CUDA 12.1 via micromamba..."
  micromamba install -y \
    -c pytorch -c nvidia -c conda-forge \
    --channel-priority flexible \
    pytorch torchvision pytorch-cuda=12.1
}

install_pytorch_pip() {
  echo "[1/2] Installing PyTorch + CUDA 12.1 via pip (fallback)..."
  pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.6.0" \
    "torchvision==0.21.0"
}

if ! install_pytorch_micromamba; then
  echo ""
  echo "micromamba failed (often missing mkl/blas channels on worker nodes)."
  echo "Trying pip instead (safe on HTCondor execution nodes with 16GB RAM)."
  echo ""
  install_pytorch_pip
fi

echo ""
echo "[2/2] Installing Python dependencies via pip (see requirements-gpu.txt)..."
pip install --no-cache-dir \
  "transformers>=4.46.0" \
  "accelerate>=0.34.0" \
  "huggingface_hub>=0.26.0" \
  "sentencepiece>=0.2.0" \
  "protobuf>=4.25.0" \
  "safetensors>=0.4.0" \
  "pillow>=10.0.0" \
  "pyyaml>=6.0" \
  "qwen-vl-utils>=0.0.8"


echo ""
echo "Done at: $(date)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "from PIL import Image; import yaml; import qwen_vl_utils; print('qwen_vl_utils ok')"
