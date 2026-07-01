#!/usr/bin/env bash
# Submit GPU pipeline job to HTCondor. Run on submit node.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/pipeline.generated.sub"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${PROJECT_ROOT}/scripts/condor_run_pipeline.sh
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/pipeline.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/pipeline.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/pipeline.\$(ClusterId).log

request_cpus    = 4
request_memory  = 48GB
request_gpus    = 1
request_runtime = 2 * 60 * 60

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT

queue
EOF

chmod +x "${PROJECT_ROOT}/scripts/condor_run_pipeline.sh"
condor_submit "${SUB_FILE}"
