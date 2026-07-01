#!/usr/bin/env bash
# Create micromamba env; install GPU deps (light step on login, heavy step via install_deps.sh).
# Usage:
#   bash scripts/setup.sh              # env only on login; run install_deps.sh after
#   bash scripts/setup.sh --install    # also run pip install (may OOM on login node)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="affordance_benchmark"
MAMBA_BIN="${HOME}/bin/micromamba"
DO_INSTALL=false

if [[ "${1:-}" == "--install" ]]; then
  DO_INSTALL=true
fi

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

if [[ "${DO_INSTALL}" == true ]]; then
  echo ""
  echo "WARNING: pip install of PyTorch on login nodes often gets OOM-killed."
  echo "Use: bash scripts/install_deps.sh  (micromamba)"
  echo "Or on submit node: bash scripts/condor_submit_install.sh"
  echo ""
  bash "${PROJECT_ROOT}/scripts/install_deps.sh"
else
  echo ""
  echo "Environment created. Install GPU packages next (pick one):"
  echo ""
  echo "  On submit node (HTCondor, recommended):"
  echo "    ssh submit"
  echo "    bash scripts/condor_submit_install.sh"
  echo "    condor_q"
  echo ""
  echo "  On login node (micromamba for PyTorch):"
  echo "    bash scripts/install_deps.sh"
  exit 0
fi

echo ""
echo "Environment ready."
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

echo ""
echo "Each new SSH session:"
echo "  source scripts/activate_env.sh"
