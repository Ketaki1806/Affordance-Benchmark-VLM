#!/usr/bin/env bash
# Avoid /tmp on login nodes (often on a full local disk).
# All temp/cache paths go to nethome (/nethome/...).
# Usage: source scripts/cluster_paths.sh

STORAGE_ROOT="${SCRATCH:-${HOME}}"

export MAMBA_ROOT_PREFIX="${STORAGE_ROOT}/micromamba"
export PIP_CACHE_DIR="${STORAGE_ROOT}/.cache/pip"
export HF_HOME="${STORAGE_ROOT}/huggingface_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TMPDIR="${STORAGE_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export TMUX_TMPDIR="${TMPDIR}"

mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${TMPDIR}"
echo "Storage root: ${STORAGE_ROOT}"
echo "TMPDIR:       ${TMPDIR}  (not /tmp — login node local disk is full)"
