#!/usr/bin/env bash
# Submit train_500 Qwen captioning to Condor (GPU).
#
# Usage (on submit):
#   bash scripts/condor_submit_train_captions.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/train_captions.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_train_captions.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" "${PROJECT_ROOT}/scripts/run_train_captions.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/train_captions.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/train_captions.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/train_captions.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 32GB
request_gpus    = 1

should_transfer_files = NO

queue
EOF

echo "Submitting train caption job from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
echo "Monitor: tail -f artifacts/logs/train_captions.<ClusterId>.out"
