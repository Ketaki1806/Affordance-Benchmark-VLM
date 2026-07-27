#!/usr/bin/env bash
# Build PACO-LVIS full-val manifest (one preferred part per unique image).
#
# Usage:
#   source scripts/activate_env.sh
#   bash scripts/build_paco_val_manifest.sh \
#     --image-root /path/to/coco \
#     --require-image \
#     --download-missing

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v python >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/activate_env.sh"
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found. Run: source scripts/activate_env.sh" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src"

python "${PROJECT_ROOT}/src/build_paco_val_manifest.py" "$@"
