#!/usr/bin/env bash
# Submit Open-VLJEPA fine-tune on train_1000 captions.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/finetune_vljepa_1000.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_finetune_open_vljepa_1000.sh"
FILTERED="${PROJECT_ROOT}/artifacts/captions/train_1000/filtered.json"

if [[ ! -f "${FILTERED}" ]]; then
  echo "ERROR: missing ${FILTERED}" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" "${PROJECT_ROOT}/scripts/run_finetune_open_vljepa_1000.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/finetune_vljepa_1000.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/finetune_vljepa_1000.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/finetune_vljepa_1000.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 32GB
request_gpus    = 1
+MaxRuntime     = 86400

should_transfer_files = NO

queue
EOF

echo "Submitting Open-VLJEPA FT train_1000 from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
