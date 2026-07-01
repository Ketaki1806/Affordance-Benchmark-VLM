#!/usr/bin/env bash
# Submit dependency install to LST HTCondor (execution node — more RAM than login).
#
# Usage (on submit node):
#   ssh submit
#   cd ~/Affordance-Benchmark-VLM
#   bash scripts/condor_submit_install.sh
#
# Monitor:
#   condor_q kahadnurkar
#   tail -f artifacts/logs/install-deps.<ClusterId>.out
#
# LST wiki: https://wiki.lst.uni-saarland.de/doku.php?id=user:cluster:a_condor
#
# Notes:
#   - should_transfer_files = NO — nethome is NFS-visible on workers; do not sandbox-copy.
#   - Do NOT use request_runtime (LST nodes lack TARGET.runtime).
#   - No request_gpus (CPU/RAM job only).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/install-deps.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_install_deps.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/install-deps.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/install-deps.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/install-deps.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 16GB

should_transfer_files = NO

queue
EOF

echo "Submitting install job from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
echo ""
echo "Monitor:"
echo "  condor_q kahadnurkar"
echo "  tail -f ${PROJECT_ROOT}/artifacts/logs/install-deps.<ClusterId>.out"
