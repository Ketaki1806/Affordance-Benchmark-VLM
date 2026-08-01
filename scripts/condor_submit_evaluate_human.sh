#!/usr/bin/env bash
# Submit human-pilot FT evaluation (CLIP-FT + Open-VLJEPA-FT on N=20).
#
# Prerequisites on cluster:
#   - humaneval/26jul/human_filtered.json
#   - artifacts/checkpoints/clip/finetuned_affordance_ep5.pt
#   - artifacts/checkpoints/open-vljepa/finetuned_affordance_ep5.pt
#   - pilot images under paths in human_filtered.json
#
# Usage (on submit):
#   bash scripts/condor_submit_evaluate_human.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_FILE="${PROJECT_ROOT}/artifacts/logs/evaluate_human.generated.sub"
WRAPPER="${PROJECT_ROOT}/scripts/condor_run_evaluate_human.sh"
HUMAN_CAPS="${PROJECT_ROOT}/humaneval/26jul/human_filtered.json"

if [[ ! -f "${HUMAN_CAPS}" ]]; then
  echo "ERROR: missing ${HUMAN_CAPS}"
  echo "From your PC:"
  echo "  scp -r humaneval/26jul kahadnurkar@login:.../Affordance-Benchmark-VLM/humaneval/"
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
echo "Monitor: tail -f artifacts/logs/evaluate_human.<ClusterId>.out"
echo "Outputs:"
echo "  artifacts/eval/pilot_human/clip_ft.json"
echo "  artifacts/eval/pilot_human/open_vljepa_ft.json"
echo "  artifacts/eval/pilot_human/summary_ft.json"
