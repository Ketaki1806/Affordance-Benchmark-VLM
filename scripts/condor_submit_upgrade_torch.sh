#!/usr/bin/env bash
# Submit torch 2.6 upgrade to Condor (do not run pip on login interactively).
#
# Usage (on submit node):
#   bash scripts/condor_submit_upgrade_torch.sh
#
# Monitor:
#   condor_q kahadnurkar
#   tail -f artifacts/logs/upgrade_torch.<ClusterId>.out

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/upgrade_torch.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_upgrade_torch.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" "${PROJECT_ROOT}/scripts/upgrade_torch.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/upgrade_torch.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/upgrade_torch.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/upgrade_torch.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 16GB
request_gpus    = 0

should_transfer_files = NO

queue
EOF

echo "Submitting torch upgrade from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
echo ""
echo "Monitor:"
echo "  condor_q kahadnurkar"
echo "  tail -f ${PROJECT_ROOT}/artifacts/logs/upgrade_torch.<ClusterId>.out"
