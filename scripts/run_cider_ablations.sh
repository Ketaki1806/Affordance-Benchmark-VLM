#!/usr/bin/env bash
# Compute CIDEr ablations for N=100 eval pairs + human-eval pilot captions.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v python >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/activate_env.sh"
fi

export PYTHONPATH="${PROJECT_ROOT}/src"

pip show pycocoevalcap >/dev/null 2>&1 || pip install --no-cache-dir pycocoevalcap

python "${PROJECT_ROOT}/src/compute_cider.py" \
  --eval-json "${1:-humaneval/30jul/clip.json}" \
  --raw-captions "${2:-humaneval/26jul/filtered.json}" \
  --human-captions "${3:-humaneval/26jul/human_filtered.json}" \
  --output "${4:-artifacts/eval/cider_ablations.json}"
