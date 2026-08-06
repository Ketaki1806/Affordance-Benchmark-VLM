#!/usr/bin/env bash
# Submit CPU job to build data/paco/manifest_train_1000.json.
# Parsing paco_lvis_v1_train.json needs more RAM than a casual submit-shell session.
#
# Usage (on submit):
#   bash scripts/condor_submit_build_train_1000_manifest.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/build_train_1000_manifest.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_build_train_1000_manifest.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" \
  "${PROJECT_ROOT}/scripts/run_build_train_1000_manifest.sh" \
  "${PROJECT_ROOT}/scripts/build_paco_train_1000.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/build_train_1000_manifest.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/build_train_1000_manifest.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/build_train_1000_manifest.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 16GB
request_gpus    = 0

should_transfer_files = NO

queue
EOF

echo "Submitting train_1000 manifest build from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
echo "tail -f artifacts/logs/build_train_1000_manifest.<ClusterId>.out"
