#!/usr/bin/env bash
# Submit a CPU Condor job to download best.pt (avoids login/submit interactive OOM).
#
# Usage (on submit node):
#   bash scripts/condor_submit_download_vljepa_ckpt.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/download_vljepa_ckpt.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_download_vljepa_ckpt.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" "${PROJECT_ROOT}/scripts/download_open_vljepa_ckpt.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/download_vljepa_ckpt.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/download_vljepa_ckpt.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/download_vljepa_ckpt.\$(ClusterId).log

getenv          = True
request_cpus    = 1
request_memory  = 4GB
request_gpus    = 0

should_transfer_files = NO

queue
EOF

echo "Submitting Open-VLJEPA checkpoint download from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
