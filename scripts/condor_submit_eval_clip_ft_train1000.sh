#!/usr/bin/env bash
# Submit eval of CLIP-FT (train_1000) on held-out val N=100.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/eval_clip_ft_train1000.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_eval_clip_ft_train1000.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/logs"
chmod +x "${WRAPPER}" "${PROJECT_ROOT}/scripts/run_eval_clip_ft_train1000.sh"

cat > "${SUB_FILE}" <<EOF
universe        = vanilla
executable      = ${WRAPPER}
initialdir      = ${PROJECT_ROOT}

output          = ${PROJECT_ROOT}/artifacts/logs/eval_clip_ft_train1000.\$(ClusterId).out
error           = ${PROJECT_ROOT}/artifacts/logs/eval_clip_ft_train1000.\$(ClusterId).err
log             = ${PROJECT_ROOT}/artifacts/logs/eval_clip_ft_train1000.\$(ClusterId).log

getenv          = True
request_cpus    = 2
request_memory  = 24GB
request_gpus    = 1

should_transfer_files = NO

queue
EOF

echo "Submitting CLIP-FT train1000 eval from: ${PROJECT_ROOT}"
condor_submit "${SUB_FILE}"
