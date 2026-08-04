#!/usr/bin/env bash
# SigLIP-only N=100: occlusion attribution + embedding modality gap.
# Run on submit after git pull (needs SigLIP encode/score fix).
#
# Usage:
#   ssh submit
#   cd ~/Affordance-Benchmark-VLM
#   bash scripts/condor_submit_siglip_n100.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_siglip_n100.sh"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/siglip_n100.generated.sub"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" \
  "${PROJECT_ROOT}/scripts/run_siglip_n100.sh" 2>/dev/null || true
chmod +x "${WRAPPER}"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/siglip_n100.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/siglip_n100.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/siglip_n100.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 32GB
request_gpus    = 1
+MaxRuntime     = 86400

should_transfer_files = NO

queue
EOF

echo "Submitting SigLIP N=100 job from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
