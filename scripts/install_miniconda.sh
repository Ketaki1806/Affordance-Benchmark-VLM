#!/usr/bin/env bash
# Install Miniconda into ~/miniconda3 (one-time setup when cluster has no conda module).
#
# WARNING: Login nodes often OOM-kill this installer. If you see "Killed", use:
#   bash scripts/install_micromamba.sh
# Or retry on a compute node with more memory:
#   srun --mem=16G --time=00:30:00 --pty bash
#   bash scripts/install_miniconda.sh
#
# Usage: bash scripts/install_miniconda.sh

set -euo pipefail

INSTALL_DIR="${HOME}/miniconda3"
INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"

if [[ -x "${INSTALL_DIR}/bin/conda" ]]; then
  echo "Miniconda already installed at ${INSTALL_DIR}"
  exit 0
fi

echo "Downloading Miniconda..."
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL -o "${INSTALLER}" "${MINICONDA_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${INSTALLER}" "${MINICONDA_URL}"
else
  echo "ERROR: neither curl nor wget found. Install Miniconda manually or load a conda module."
  exit 1
fi
echo "Installing to ${INSTALL_DIR}..."
bash "${INSTALLER}" -b -p "${INSTALL_DIR}"
rm -f "${INSTALLER}"

# shellcheck disable=SC1091
source "${INSTALL_DIR}/etc/profile.d/conda.sh"

echo ""
echo "Miniconda installed. Activate conda in this shell with:"
echo "  source ~/miniconda3/etc/profile.d/conda.sh"
echo ""
echo "Then run:"
echo "  bash scripts/setup_env.sh"
