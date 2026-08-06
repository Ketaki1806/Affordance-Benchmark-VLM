#!/usr/bin/env bash
# After train_1000 CLIP FT: re-score held-out val N=100 pairs with the FT checkpoint.
#
# Usage:
#   bash scripts/run_eval_clip_ft_train1000.sh
#   PAIRS_JSON=humaneval/30jul/clip.json bash scripts/run_eval_clip_ft_train1000.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${HF_HOME}"

CFG="${CFG:-${PROJECT_ROOT}/configs/config_eval_clip_ft_train1000.yaml}"
PAIRS_JSON="${PAIRS_JSON:-${PROJECT_ROOT}/humaneval/30jul/clip.json}"
FILTERED="${PROJECT_ROOT}/artifacts/captions/val_100_pairs/filtered.json"
CKPT="${PROJECT_ROOT}/artifacts/checkpoints/clip/finetuned_affordance_train1000_ep5.pt"

if [[ ! -f "${CKPT}" ]]; then
  # fall back to non-ep name if only best was written
  ALT="${PROJECT_ROOT}/artifacts/checkpoints/clip/finetuned_affordance_train1000.pt"
  if [[ -f "${ALT}" ]]; then
    echo "NOTE: using ${ALT} (ep5 file missing)"
    # rewrite eval by passing through config that still points at ep5 — user may need to symlink
    ln -sfn "$(basename "${ALT}")" "${CKPT}" || cp "${ALT}" "${CKPT}"
  else
    echo "ERROR: missing FT checkpoint ${CKPT}" >&2
    exit 1
  fi
fi

if [[ ! -f "${FILTERED}" ]]; then
  if [[ ! -f "${PAIRS_JSON}" ]]; then
    echo "ERROR: need ${PAIRS_JSON} or ${FILTERED}" >&2
    exit 1
  fi
  python "${PROJECT_ROOT}/src/pairs_to_filtered.py" \
    --pairs-json "${PAIRS_JSON}" \
    --out "${FILTERED}"
fi

python "${PROJECT_ROOT}/scripts/verify_gpu.py" || true
LOG_FILE="${PROJECT_ROOT}/artifacts/logs/eval-clip-ft-train1000-$(date +%Y%m%d-%H%M%S).log"
echo "Eval CLIP-FT (train_1000) on val N=100. Log: ${LOG_FILE}"

python "${PROJECT_ROOT}/src/evaluate.py" \
  --config "${CFG}" \
  --filtered "${FILTERED}" \
  --backends clip \
  2>&1 | tee "${LOG_FILE}"
