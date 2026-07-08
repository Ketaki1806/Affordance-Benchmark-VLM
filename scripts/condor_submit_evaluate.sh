#!/usr/bin/env bash
# Submit GPU evaluation job to LST HTCondor. Run on submit node.
#
# Usage:
#   ssh submit
#   cd ~/Affordance-Benchmark-VLM
#   bash scripts/condor_submit_evaluate.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/evaluate.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_evaluate.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/evaluate.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/evaluate.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/evaluate.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 32GB
request_gpus    = 1

should_transfer_files = NO

queue
EOF

echo "Submitting evaluation job from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
