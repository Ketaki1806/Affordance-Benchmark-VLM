#!/usr/bin/env bash
# Submit install job to HTCondor. Run from login, then: ssh submit && bash scripts/condor_submit_install.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/install-deps.generated.sub"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${PROJECT_ROOT}/scripts/condor_install_deps.sh
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/install-deps.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/install-deps.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/install-deps.\$(ClusterId).log

request_cpus    = 2
request_memory  = 16GB
request_runtime = 60 * 60

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT

queue
EOF

chmod +x "${PROJECT_ROOT}/scripts/condor_install_deps.sh"
condor_submit "${SUB_FILE}"
