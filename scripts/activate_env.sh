#!/usr/bin/env bash
# Activate affordance_benchmark via micromamba.
# Usage: source scripts/activate_env.sh

ENV_NAME="affordance_benchmark"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "${HOME}/bin/micromamba" ]]; then
  cat <<'EOF'
ERROR: micromamba not found.

  bash scripts/install_micromamba.sh
  bash scripts/setup.sh
EOF
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cluster_paths.sh"
eval "$("${HOME}/bin/micromamba" shell hook -s bash -r "${MAMBA_ROOT_PREFIX}")"
micromamba activate "${ENV_NAME}"
return 0 2>/dev/null || exit 0
