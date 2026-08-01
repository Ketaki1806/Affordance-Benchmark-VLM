#!/usr/bin/env bash
# Condor: Open-VLJEPA fine-tune (needs train_500/filtered.json).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/finetune_vljepa.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_finetune_open_vljepa.sh"
FILTERED="${PROJECT_ROOT}/artifacts/captions/train_500/filtered.json"

if [[ ! -f "${FILTERED}" ]]; then
  echo "ERROR: missing ${FILTERED}"
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" "${PROJECT_ROOT}/scripts/run_finetune_open_vljepa.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/finetune_vljepa.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/finetune_vljepa.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/finetune_vljepa.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 48GB
request_gpus    = 1

should_transfer_files = NO

queue
EOF

echo "Submitting Open-VLJEPA fine-tune from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
echo "tail -f artifacts/logs/finetune_vljepa.<ClusterId>.out"
