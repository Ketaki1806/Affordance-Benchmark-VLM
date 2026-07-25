#!/usr/bin/env bash
# Build PACO-LVIS pilot manifest on the login node.
#
# Usage:
#   source scripts/activate_env.sh
#   bash scripts/build_paco_pilot_manifest.sh --image-root data/paco/coco --download-missing --copy-images

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Prefer the micromamba env if the user forgot to activate it.
if ! command -v python >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/activate_env.sh"
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found. Run: source scripts/activate_env.sh" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src"

python "${PROJECT_ROOT}/src/build_paco_pilot_manifest.py" "$@"
