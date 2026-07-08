#!/usr/bin/env bash
# Clone Open-VLJEPA and download the MSRVTT checkpoint (one-time setup).
#
# Requires Hugging Face login with access to:
#   - meta-llama/Llama-3.2-1B
#   - google/embeddinggemma-300m
#   - cun-bjy/open-vljepa
#   - facebook/vjepa2-vitl-fpc64-256
#
# Usage:
#   bash scripts/setup_open_vljepa.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${PROJECT_ROOT}/vendor/open-vljepa"
CKPT_DIR="${PROJECT_ROOT}/artifacts/checkpoints/open-vljepa"
REPO_URL="https://github.com/dion-jy/open-vljepa.git"

cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  echo "Cloning Open-VLJEPA into ${VENDOR_DIR}..."
  mkdir -p "${PROJECT_ROOT}/vendor"
  git clone "${REPO_URL}" "${VENDOR_DIR}"
else
  echo "Open-VLJEPA repo already present: ${VENDOR_DIR}"
fi

mkdir -p "${CKPT_DIR}"

if [[ ! -f "${CKPT_DIR}/best.pt" ]]; then
  echo "Downloading checkpoint to ${CKPT_DIR}..."
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download cun-bjy/open-vljepa best.pt --local-dir "${CKPT_DIR}"
  else
    python - <<PY
from huggingface_hub import hf_hub_download
from pathlib import Path

ckpt_dir = Path("${CKPT_DIR}")
ckpt_dir.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id="cun-bjy/open-vljepa",
    filename="best.pt",
    local_dir=str(ckpt_dir),
)
print("Downloaded:", path)
PY
  fi
else
  echo "Checkpoint already present: ${CKPT_DIR}/best.pt"
fi

echo ""
echo "Open-VLJEPA setup complete."
echo "  repo:      ${VENDOR_DIR}"
echo "  checkpoint: ${CKPT_DIR}/best.pt"
echo ""
echo "Run evaluation (after caption pipeline):"
echo "  bash scripts/run_evaluate.sh"
