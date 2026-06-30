#!/usr/bin/env bash
# Make conda available on HPC clusters (module system or user install).
# Usage: source scripts/load_conda.sh

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  return 0 2>/dev/null || exit 0
fi

if command -v module >/dev/null 2>&1; then
  for mod in miniforge3 Miniforge3 miniconda3 anaconda3; do
    if module avail "${mod}" 2>&1 | grep -qi "${mod}"; then
      module load "${mod}"
      break
    fi
  done
fi

if [[ -n "${CONDASH:-}" && -f "${CONDASH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDASH}"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  cat <<'EOF'
ERROR: conda not found.

On the LST cluster, try:
  module avail 2>&1 | grep -iE 'conda|miniforge|python'
  module load miniforge3    # or miniconda3 / anaconda3 (name varies)

If no module exists, use micromamba (works on memory-limited login nodes):
  bash scripts/install_micromamba.sh
  bash scripts/setup_env_micromamba.sh
EOF
  return 1 2>/dev/null || exit 1
fi
