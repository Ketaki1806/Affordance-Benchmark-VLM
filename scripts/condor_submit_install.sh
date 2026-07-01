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
#   - Do NOT use request_runtime on LST (nodes lack TARGET.runtime → no match).
#   - Do NOT use request_disk (packages install to nethome, not worker scratch).
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

request_cpus    = 2
request_memory  = 16GB

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT

queue
EOF

echo "Submitting install job from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
echo ""
echo "Monitor:"
echo "  condor_q kahadnurkar"
echo "  condor_q kahadnurkar -better-analyze <ClusterId>"
echo "  tail -f ${PROJECT_ROOT}/artifacts/logs/install-deps.<ClusterId>.out"
