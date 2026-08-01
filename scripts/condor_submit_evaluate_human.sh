#!/usr/bin/env bash
# Condor: FT CLIP + VLJEPA on human pilot captions (N=20).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/evaluate_human.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_evaluate_human.sh"
HUMAN_CAPS="${PROJECT_ROOT}/humaneval/26jul/human_filtered.json"

if [[ ! -f "${HUMAN_CAPS}" ]]; then
  echo "ERROR: missing ${HUMAN_CAPS}"
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" "${PROJECT_ROOT}/scripts/run_evaluate_human.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/evaluate_human.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/evaluate_human.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/evaluate_human.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 32GB
request_gpus    = 1

should_transfer_files = NO

queue
EOF

echo "Submitting human-pilot FT eval from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
echo "tail -f artifacts/logs/evaluate_human.<ClusterId>.out"
