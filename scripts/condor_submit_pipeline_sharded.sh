#!/usr/bin/env bash
# Submit sharded GPU caption pipeline jobs to LST HTCondor.
#
# Usage (on submit node):
#   bash scripts/condor_submit_pipeline_sharded.sh
#   bash scripts/condor_submit_pipeline_sharded.sh 10   # 10 shards
#
# Each Process writes artifacts/captions/.../raw.shard{N}.json
# After all jobs finish:
#   export PYTHONPATH=src
#   python src/merge_caption_shards.py

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_SHARDS="${1:-10}"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/pipeline_sharded.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_pipeline_sharded.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
arguments       = \$(Process) ${NUM_SHARDS}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/pipeline_shard.\$(ClusterId).\$(Process).out
error           = ${PROJECT_ROOT}/artifacts/logs/pipeline_shard.\$(ClusterId).\$(Process).err
log             = ${PROJECT_ROOT}/artifacts/logs/pipeline_shard.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 32GB
request_gpus    = 1

should_transfer_files = NO

queue ${NUM_SHARDS}
EOF

echo "Submitting ${NUM_SHARDS} sharded pipeline jobs from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
echo "After completion, merge with:"
echo "  source scripts/activate_env.sh && export PYTHONPATH=src && python src/merge_caption_shards.py"
