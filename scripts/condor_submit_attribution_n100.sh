#!/usr/bin/env bash
# Submit N=100 GPU occlusion attribution job to LST HTCondor. Run on submit node.
#
# Usage:
#   ssh submit
#   cd ~/Affordance-Benchmark-VLM
#   bash scripts/condor_submit_attribution_n100.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/attribution_n100.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_attribution_n100.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" "${PROJECT_ROOT}/scripts/run_attribution_n100.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/attribution_n100.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/attribution_n100.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/attribution_n100.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 32GB
request_gpus    = 1

# Full N=100 × 3 backends can run several hours
+MaxRuntime     = 86400

should_transfer_files = NO

queue
EOF

echo "Submitting N=100 occlusion attribution job from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
