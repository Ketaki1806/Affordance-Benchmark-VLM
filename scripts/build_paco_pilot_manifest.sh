#!/usr/bin/env bash
# Build PACO-LVIS pilot manifest on the login node.
#
# Usage:
#   source scripts/activate_env.sh
#   bash scripts/build_paco_pilot_manifest.sh --image-root /path/to/coco --require-image --copy-images

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src"

python "${PROJECT_ROOT}/src/build_paco_pilot_manifest.py" "$@"
