#!/usr/bin/env bash
# Caption train_1000 with Qwen (single GPU). Prefer the sharded Condor submit for speed.
#
# Usage:
#   bash scripts/run_train_1000_captions.sh
#   bash scripts/run_train_1000_captions.sh --shard-index 0 --num-shards 10

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${PROJECT_ROOT}/artifacts/captions/train_1000" "${HF_HOME}"

CFG="${CFG:-${PROJECT_ROOT}/configs/config_train_1000.yaml}"
MANIFEST="${PROJECT_ROOT}/data/paco/manifest_train_1000.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: missing ${MANIFEST}" >&2
  echo "Run: bash scripts/build_paco_train_1000.sh" >&2
  exit 1
fi

python "${PROJECT_ROOT}/scripts/verify_gpu.py" || true

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/pipeline-train1000-$(date +%Y%m%d-%H%M%S).log"
echo "train_1000 captions. Config: ${CFG}"
echo "Log: ${LOG_FILE}"

python "${PROJECT_ROOT}/src/pipeline.py" \
  --config "${CFG}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
