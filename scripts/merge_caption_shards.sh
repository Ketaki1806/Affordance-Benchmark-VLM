#!/usr/bin/env bash
# Merge sharded caption outputs into raw.json + filtered.json (paths from config).
#
# Usage:
#   source scripts/activate_env.sh
#   bash scripts/merge_caption_shards.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v python >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/activate_env.sh"
fi

export PYTHONPATH="${PROJECT_ROOT}/src"

python "${PROJECT_ROOT}/src/merge_caption_shards.py" "$@"
