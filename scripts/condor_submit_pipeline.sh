#!/usr/bin/env bash
# Submit GPU pipeline job to LST HTCondor. Run on submit node.
#
# Usage:
#   ssh submit
#   cd ~/Affordance-Benchmark-VLM
#   bash scripts/condor_submit_pipeline.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/pipeline.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_pipeline.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/pipeline.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/pipeline.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/pipeline.\$(ClusterId).log

getenv          = True
request_cpus    = 4
request_memory  = 48GB
request_gpus    = 1

should_transfer_files = NO

queue
EOF

echo "Submitting pipeline job from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
