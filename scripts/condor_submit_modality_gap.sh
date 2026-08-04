#!/usr/bin/env bash
# Submit embedding modality gap job (cheap; can share queue with attribution_n100).
#
# Usage:
#   bash scripts/condor_submit_modality_gap.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/modality_gap.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_modality_gap.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" "${PROJECT_ROOT}/scripts/run_modality_gap.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/modality_gap.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/modality_gap.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/modality_gap.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 24GB
request_gpus    = 1

should_transfer_files = NO

queue
EOF

echo "Submitting embedding modality gap job from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
