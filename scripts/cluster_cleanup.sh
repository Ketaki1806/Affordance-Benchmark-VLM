#!/usr/bin/env bash
# Remove cluster install artifacts from nethome (micromamba, caches, temp, etc.).
# Does NOT delete the project repo unless you pass --project.
#
# Usage:
#   bash scripts/cluster_cleanup.sh              # env + caches + temp
#   bash scripts/cluster_cleanup.sh --bashrc     # also fix ~/.bashrc
#   bash scripts/cluster_cleanup.sh --artifacts  # also clear artifacts/logs
#   bash scripts/cluster_cleanup.sh --all        # everything except repo
#   bash scripts/cluster_cleanup.sh --project    # also delete project directory

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="affordance_benchmark"

FIX_BASHRC=false
CLEAN_ARTIFACTS=false
CLEAN_PROJECT=false

for arg in "$@"; do
  case "${arg}" in
    --bashrc) FIX_BASHRC=true ;;
    --artifacts) CLEAN_ARTIFACTS=true ;;
    --project) CLEAN_PROJECT=true ;;
    --all)
      FIX_BASHRC=true
      CLEAN_ARTIFACTS=true
      ;;
    -h|--help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}  (try --help)"
      exit 1
      ;;
  esac
done

echo "=== Disk usage before ==="
du -sh "${HOME}" 2>/dev/null || true
quota -s 2>/dev/null || true

echo ""
echo "=== Stopping background installs ==="
pkill -f "pip install.*requirements-gpu" 2>/dev/null || true
pkill -f ".run_pip_install.sh" 2>/dev/null || true
pkill -f "micromamba install.*pytorch" 2>/dev/null || true

if [[ -x "${HOME}/bin/micromamba" ]]; then
  echo ""
  echo "=== Removing micromamba environment ==="
  export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
  "${HOME}/bin/micromamba" env remove -n "${ENV_NAME}" -y 2>/dev/null || true
fi

echo ""
echo "=== Removing environments and tools ==="
rm -rf "${HOME}/micromamba"
rm -rf "${HOME}/miniconda3" "${HOME}/miniforge3"
rm -f "${HOME}/bin/micromamba"

echo ""
echo "=== Removing caches and temp (nethome) ==="
rm -rf "${HOME}/.cache/pip" "${HOME}/.cache/conda" "${HOME}/.cache/huggingface"
rm -rf "${HOME}/huggingface_cache"
rm -rf "${HOME}/tmp"
rm -f "${HOME}/.run_pip_install.sh" "${HOME}/pip_install.log"

if [[ "${CLEAN_ARTIFACTS}" == true ]]; then
  echo ""
  echo "=== Removing project artifacts ==="
  rm -rf "${PROJECT_ROOT}/artifacts"
fi

if [[ "${FIX_BASHRC}" == true && -f "${HOME}/.bashrc" ]]; then
  echo ""
  echo "=== Cleaning ~/.bashrc ==="
  cp "${HOME}/.bashrc" "${HOME}/.bashrc.bak.$(date +%Y%m%d-%H%M%S)"
  sed -i \
    -e '/^Running `shell init`/d' \
    -e '/^ - modifies RC file/d' \
    -e '/^ - generates config for root prefix/d' \
    -e '/^ - sets mamba executable to/d' \
    -e '/^The following has been added in your/d' \
    "${HOME}/.bashrc"
  # Remove all micromamba init blocks (activate_env.sh does not need them)
  sed -i '/# >>> mamba initialize >>>/,/# <<< mamba initialize <<</d' "${HOME}/.bashrc"
  echo "Backup saved; micromamba blocks removed from ~/.bashrc"
fi

if [[ "${CLEAN_PROJECT}" == true ]]; then
  echo ""
  echo "=== Removing project directory ==="
  echo "Deleting: ${PROJECT_ROOT}"
  rm -rf "${PROJECT_ROOT}"
fi

echo ""
echo "=== Disk usage after ==="
du -sh "${HOME}" 2>/dev/null || true
quota -s 2>/dev/null || true

echo ""
if [[ "${CLEAN_PROJECT}" == true ]]; then
  echo "Done. Project and install artifacts removed."
  echo "Re-clone: git clone <repo-url> ~/Affordance_Benchmark"
else
  echo "Done. Install artifacts removed; repo kept at: ${PROJECT_ROOT}"
  echo "Fresh setup:"
  echo "  bash scripts/install_micromamba.sh"
  echo "  bash scripts/setup.sh"
  echo "  bash scripts/install_deps.sh"
fi
