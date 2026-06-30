#!/usr/bin/env bash
# Use scratch/work storage on HPC (home dirs are often tiny).
# Usage: source scripts/cluster_paths.sh

# Try common scratch locations (set SCRATCH manually if your cluster uses another path)
if [[ -z "${SCRATCH:-}" ]]; then
  for candidate in \
    "${HOME}/scratch" \
    "/scratch/${USER}" \
    "/work/${USER}" \
    "/tmp/${USER}"; do
    if [[ -d "${candidate}" ]] && [[ -w "${candidate}" ]]; then
      SCRATCH="${candidate}"
      break
    fi
  done
fi

if [[ -n "${SCRATCH:-}" ]]; then
  export MAMBA_ROOT_PREFIX="${SCRATCH}/micromamba"
  export PIP_CACHE_DIR="${SCRATCH}/.cache/pip"
  export HF_HOME="${SCRATCH}/huggingface_cache"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  export TMPDIR="${SCRATCH}/tmp"
  mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${TMPDIR}"
  echo "Using scratch storage: ${SCRATCH}"
else
  echo "WARNING: No scratch dir found. Using home (may run out of space)."
  export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
  export PIP_CACHE_DIR="${HOME}/.cache/pip"
  export HF_HOME="${HOME}/huggingface_cache"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
fi
