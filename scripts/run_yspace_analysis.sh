#!/usr/bin/env bash
# Run EmbeddingGemma Y-space caption confusability analysis.
#
# Usage:
#   source scripts/activate_env.sh
#   bash scripts/run_yspace_analysis.sh
#   # optional: huggingface-cli login  (if embeddinggemma is gated)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v python >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/activate_env.sh"
fi

export PYTHONPATH="${PROJECT_ROOT}/src"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh" 2>/dev/null || true

python "${PROJECT_ROOT}/src/analyze_caption_yspace.py" "$@"
