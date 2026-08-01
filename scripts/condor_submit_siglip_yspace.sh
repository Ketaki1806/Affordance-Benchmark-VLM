#!/usr/bin/env bash
# Condor: SigLIP eval + EmbeddingGemma Y-space (N=100 filtered captions).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/siglip_yspace.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_siglip_yspace.sh"
FILTERED="${PROJECT_ROOT}/artifacts/captions/val_full/filtered.json"

if [[ ! -f "${FILTERED}" ]]; then
  echo "ERROR: missing ${FILTERED}"
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" "${PROJECT_ROOT}/scripts/run_siglip_yspace.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/siglip_yspace.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/siglip_yspace.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/siglip_yspace.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 32GB
request_gpus    = 1

should_transfer_files = NO

queue
EOF

echo "Submitting SigLIP + Y-space from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
echo "tail -f artifacts/logs/siglip_yspace.<ClusterId>.out"
