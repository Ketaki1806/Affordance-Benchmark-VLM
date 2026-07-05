#!/usr/bin/env bash
# Probe LST HTCondor pool and suggest request_cpus / request_memory / request_gpus.
#
# Run on the submit node:
#   ssh submit
#   cd ~/Affordance-Benchmark-VLM
#   bash scripts/condor_check_resources.sh
#
# Options:
#   --probe   Submit a short-lived test job and run condor_q -better-analyze (optional)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DO_PROBE=false

for arg in "$@"; do
  case "${arg}" in
    --probe) DO_PROBE=true ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}  (try --help)"
      exit 1
      ;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: '$1' not found. Run this on the submit node: ssh submit"
    exit 1
  fi
}

require_cmd condor_status
require_cmd condor_q

echo "=== HTCondor resource probe ==="
echo "Host:    $(hostname)"
echo "User:    $(whoami)"
echo "Project: ${PROJECT_ROOT}"
echo ""

echo "=== Queue summary ==="
condor_q -totals 2>/dev/null || condor_q
echo ""

pick_gpu_attr() {
  local attr
  for attr in TotalGPUs DetectedGPUs GPUs Gpus; do
    if condor_status -af "${attr}" 2>/dev/null | grep -qv '^$'; then
      echo "${attr}"
      return 0
    fi
  done
  return 1
}

GPU_ATTR=""
if GPU_ATTR="$(pick_gpu_attr)"; then
  echo "GPU attribute detected: ${GPU_ATTR}"
else
  echo "WARNING: No standard GPU attribute found on slots (TotalGPUs/DetectedGPUs/GPUs)."
  GPU_ATTR="TotalGPUs"
fi
echo ""

count_slots() {
  local constraint="$1"
  condor_status -const "${constraint}" 2>/dev/null | grep -c '^[[:alnum:]]' || echo 0
}

show_slot_sample() {
  local label="$1"
  local constraint="$2"
  echo "--- ${label} (matching slots: $(count_slots "${constraint}")) ---"
  condor_status -const "${constraint}" -af Name Cpus Memory "${GPU_ATTR}" State Activity 2>/dev/null \
    | head -15 || echo "(none or attribute unavailable)"
  echo ""
}

echo "=== Sample slots (first 15 rows each) ==="
show_slot_sample "All unclaimed slots" 'State == "Unclaimed" || Activity == "Idle"'
show_slot_sample "GPU slots (${GPU_ATTR} >= 1)" "${GPU_ATTR} >= 1"
show_slot_sample "Large CPU slots (Cpus>=2, Memory>=16384)" 'Cpus >= 2 && Memory >= 16384'
echo ""

echo "=== Match counts for install job (no GPU) ==="
printf "%-28s %s\n" "request_cpus=1, memory=8GB"  "$(count_slots 'Cpus >= 1 && Memory >= 8192')"
printf "%-28s %s\n" "request_cpus=2, memory=16GB" "$(count_slots 'Cpus >= 2 && Memory >= 16384')"
printf "%-28s %s\n" "request_cpus=4, memory=16GB" "$(count_slots 'Cpus >= 4 && Memory >= 16384')"
echo ""

echo "=== Match counts for pipeline job (1 GPU, Qwen2.5-7B) ==="
printf "%-36s %s\n" "gpus=1, cpus=2, memory=24GB" \
  "$(count_slots "${GPU_ATTR} >= 1 && Cpus >= 2 && Memory >= 24576")"
printf "%-36s %s\n" "gpus=1, cpus=2, memory=32GB" \
  "$(count_slots "${GPU_ATTR} >= 1 && Cpus >= 2 && Memory >= 32768")"
printf "%-36s %s\n" "gpus=1, cpus=4, memory=32GB" \
  "$(count_slots "${GPU_ATTR} >= 1 && Cpus >= 4 && Memory >= 32768")"
printf "%-36s %s\n" "gpus=1, cpus=4, memory=48GB (old)" \
  "$(count_slots "${GPU_ATTR} >= 1 && Cpus >= 4 && Memory >= 49152")"
echo ""

# Pick recommendation: highest match count among pipeline configs
best_label=""
best_count=-1
for cfg in \
  "2|24576|gpus=1, cpus=2, memory=24GB" \
  "2|32768|gpus=1, cpus=2, memory=32GB" \
  "4|32768|gpus=1, cpus=4, memory=32GB" \
  "4|49152|gpus=1, cpus=4, memory=48GB"; do
  cpus="${cfg%%|*}"; rest="${cfg#*|}"
  mem="${rest%%|*}"; label="${rest#*|}"
  n="$(count_slots "${GPU_ATTR} >= 1 && Cpus >= ${cpus} && Memory >= ${mem}")"
  if [[ "${n}" -gt "${best_count}" ]]; then
    best_count="${n}"
    best_cpus="${cpus}"
    best_mem_gb=$((mem / 1024))
    best_label="${label}"
  fi
done

echo "=== Recommended submit settings ==="
echo ""
echo "Install (scripts/condor_submit_install.sh):"
echo "  request_cpus   = 2"
echo "  request_memory = 16GB"
echo "  (no request_gpus)"
echo ""
echo "Pipeline (scripts/condor_submit_pipeline.sh):"
if [[ "${best_count}" -gt 0 ]]; then
  echo "  request_cpus   = ${best_cpus}"
  echo "  request_memory = ${best_mem_gb}GB"
  echo "  request_gpus   = 1"
  echo "  # best match among tested configs: ${best_label} (${best_count} slots)"
else
  echo "  request_cpus   = 2"
  echo "  request_memory = 32GB"
  echo "  request_gpus   = 1"
  echo "  # no GPU slots matched probes; check condor_status or LST wiki"
fi
echo ""
echo "Always use:"
echo "  getenv = True"
echo "  should_transfer_files = NO"
echo "  # do NOT use request_runtime on LST"
echo ""

if [[ "${DO_PROBE}" == true ]]; then
  require_cmd condor_submit
  PROBE="${PROJECT_ROOT}/artifacts/logs/probe.generated.sub"
  mkdir -p "${PROJECT_ROOT}/artifacts/logs"
  cat > "${PROBE}" <<EOF
universe   = vanilla
executable = /bin/true
initialdir = ${PROJECT_ROOT}
output     = ${PROJECT_ROOT}/artifacts/logs/probe.\$(ClusterId).out
error      = ${PROJECT_ROOT}/artifacts/logs/probe.\$(ClusterId).err
log        = ${PROJECT_ROOT}/artifacts/logs/probe.\$(ClusterId).log
getenv     = True
request_cpus   = ${best_cpus:-2}
request_memory = ${best_mem_gb:-32}GB
request_gpus   = 1
should_transfer_files = NO
queue
EOF
  echo "=== Probe job (exits immediately) ==="
  JOB_LINE="$(condor_submit "${PROBE}")"
  echo "${JOB_LINE}"
  CLUSTER="$(echo "${JOB_LINE}" | awk '{print $6}' | tr -d '.')"
  sleep 2
  condor_q "${USER}" -better-analyze "${CLUSTER}" 2>/dev/null || condor_q -better-analyze "${CLUSTER}"
  echo ""
  echo "Remove probe: condor_rm ${CLUSTER}"
fi
