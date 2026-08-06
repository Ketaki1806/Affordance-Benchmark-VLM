#!/usr/bin/env bash
# Submit sharded Qwen captioning for train_1000 (default 10 GPU shards).
#
# Usage (on submit):
#   bash scripts/build_paco_train_1000.sh          # once
#   bash scripts/condor_submit_train_1000_captions_sharded.sh
#   bash scripts/condor_submit_train_1000_captions_sharded.sh 5   # 5 shards
#
# After all shards finish:
#   source scripts/activate_env.sh && export PYTHONPATH=src
#   python src/merge_caption_shards.py --config configs/config_train_1000.yaml

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_SHARDS="${1:-10}"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/train_1000_captions_sharded.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_train_1000_captions_sharded.sh"
MANIFEST="${PROJECT_ROOT}/data/paco/manifest_train_1000.json"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: missing ${MANIFEST}" >&2
  echo "Run: bash scripts/build_paco_train_1000.sh" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/artifacts/logs" "${PROJECT_ROOT}/artifacts/captions/train_1000"
chmod +x "${WRAPPER}" "${PROJECT_ROOT}/scripts/run_train_1000_captions.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
arguments       = \$(Process) ${NUM_SHARDS}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/train_1000_cap.\$(ClusterId).\$(Process).out
error           = ${PROJECT_ROOT}/artifacts/logs/train_1000_cap.\$(ClusterId).\$(Process).err
log             = ${PROJECT_ROOT}/artifacts/logs/train_1000_cap.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 32GB
request_gpus    = 1
+MaxRuntime     = 86400

should_transfer_files = NO

queue ${NUM_SHARDS}
EOF

echo "Submitting ${NUM_SHARDS} train_1000 caption shards from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
echo "When done, merge:"
echo "  source scripts/activate_env.sh && export PYTHONPATH=src"
echo "  python src/merge_caption_shards.py --config configs/config_train_1000.yaml"
