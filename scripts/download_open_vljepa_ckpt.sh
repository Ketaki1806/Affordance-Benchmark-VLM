#!/usr/bin/env bash
# Download Open-VLJEPA best.pt with curl (low memory; survives login/submit OOM kills
# that hit `hf download` on large files).
#
# Prerequisites:
#   source scripts/activate_env.sh
#   hf auth login   # or export HF_TOKEN=hf_...
#
# Usage:
#   bash scripts/download_open_vljepa_ckpt.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${PROJECT_ROOT}/artifacts/checkpoints/open-vljepa"
DEST="${CKPT_DIR}/best.pt"
URL="https://huggingface.co/cun-bjy/open-vljepa/resolve/main/best.pt"

mkdir -p "${CKPT_DIR}" "${PROJECT_ROOT}/artifacts/logs"

if [[ -f "${DEST}" ]]; then
  size=$(stat -c%s "${DEST}" 2>/dev/null || stat -f%z "${DEST}")
  # ~2.13 GiB = 2285895680 approx; accept >= 2e9 bytes
  if [[ "${size}" -ge 2000000000 ]]; then
    echo "Already present: ${DEST} (${size} bytes)"
    exit 0
  fi
  echo "Incomplete file (${size} bytes); resuming..."
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  # Prefer token file written by `hf auth login`
  TOKEN_FILE="${HOME}/.cache/huggingface/token"
  if [[ -f "${TOKEN_FILE}" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: set HF_TOKEN or run: hf auth login" >&2
  exit 1
fi

echo "Downloading ${URL}"
echo " -> ${DEST}"
curl -L --fail --retry 5 --retry-delay 5 \
  -C - \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  -o "${DEST}" \
  "${URL}"

size=$(stat -c%s "${DEST}" 2>/dev/null || stat -f%z "${DEST}")
echo "Done: ${DEST} (${size} bytes)"
if [[ "${size}" -lt 2000000000 ]]; then
  echo "WARNING: file looks smaller than expected (~2.13G)" >&2
  exit 1
fi
