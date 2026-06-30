#!/usr/bin/env bash
# Activate affordance_benchmark env via conda or micromamba.
# Usage: source scripts/activate_env.sh

ENV_NAME="affordance_benchmark"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/load_conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/load_conda.sh" 2>/dev/null && conda activate "${ENV_NAME}" && return 0 2>/dev/null
fi

if [[ -x "${HOME}/bin/micromamba" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/cluster_paths.sh"
  eval "$("${HOME}/bin/micromamba" shell hook -s bash -r "${MAMBA_ROOT_PREFIX}")"
  micromamba activate "${ENV_NAME}"
  return 0 2>/dev/null || exit 0
fi

cat <<'EOF'
ERROR: No environment manager found.

Option 1 — cluster module:
  module avail 2>&1 | grep -iE 'conda|miniforge|python'
  module load miniforge3
  bash scripts/setup_env.sh

Option 2 — micromamba (recommended on LST login node):
  bash scripts/install_micromamba.sh
  bash scripts/setup_env_micromamba.sh
EOF
return 1 2>/dev/null || exit 1
