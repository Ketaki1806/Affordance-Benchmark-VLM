#!/usr/bin/env bash
# Lightweight conda alternative — works on login nodes where Miniconda gets OOM-killed.
# Uses a single binary download (no tar required).
# Usage: bash scripts/install_micromamba.sh

set -euo pipefail

MAMBA_ROOT="${HOME}/micromamba"
MAMBA_BIN="${HOME}/bin/micromamba"
MAMBA_URL="https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"

if [[ -x "${MAMBA_BIN}" ]]; then
  echo "Micromamba already installed at ${MAMBA_BIN}"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl not found."
  exit 1
fi

mkdir -p "${HOME}/bin"

echo "Downloading micromamba binary (no tar needed)..."
curl -fsSL -o "${MAMBA_BIN}" "${MAMBA_URL}"
chmod +x "${MAMBA_BIN}"

echo ""
echo "Micromamba installed at ${MAMBA_BIN}"
echo "Does not modify ~/.bashrc — use: source scripts/activate_env.sh"
echo ""
echo "Then run:"
echo "  bash scripts/setup.sh"
