#!/usr/bin/env bash
# Upgrade PyTorch to >=2.6 (required by EmbeddingGemma / Gemma3 masking).
#
# Usage (after activate_env.sh):
#   bash scripts/upgrade_torch.sh

set -euo pipefail

echo "Before: $(python -c 'import torch; print(torch.__version__)' 2>&1 || echo 'import failed')"

pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.6.0" \
  "torchvision==0.21.0"

python -c "import torch; print('After:', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
