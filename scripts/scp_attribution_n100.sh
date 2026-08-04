#!/usr/bin/env bash
# Pull N=100 occlusion + embedding modality gap artifacts from LST.
#
# From your laptop use the *login* host (not "submit" — that name only
# resolves inside the cluster). Nethome is shared, so artifacts written
# on submit/GPU jobs are visible at the same path on login.
#
# Usage (Git Bash / WSL / macOS / Linux):
#   bash scripts/scp_attribution_n100.sh
#   bash scripts/scp_attribution_n100.sh --with-logs
#
# PowerShell:
#   scp -r kahadnurkar@login.lst.uni-saarland.de:~/Affordance-Benchmark-VLM/artifacts/attribution_n100 ./artifacts/

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Override if needed, e.g. REMOTE=kahadnurkar@login.lst.uni-saarland.de:~/Affordance_Benchmark
REMOTE="${REMOTE:-kahadnurkar@login.lst.uni-saarland.de:~/Affordance-Benchmark-VLM}"
LOCAL_ROOT="${PROJECT_ROOT}/artifacts"
WITH_LOGS=0

for arg in "$@"; do
  case "${arg}" in
    --with-logs) WITH_LOGS=1 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: ${arg}" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOCAL_ROOT}"

echo "Remote: ${REMOTE}"
echo "Local:  ${LOCAL_ROOT}/attribution_n100"
echo ""

scp -r "${REMOTE}/artifacts/attribution_n100" "${LOCAL_ROOT}/"

echo ""
echo "Pulled attribution_n100. Key files:"
ls -la "${LOCAL_ROOT}/attribution_n100/summary.json" \
       "${LOCAL_ROOT}/attribution_n100/embedding_modality_gap.json" 2>/dev/null || true

if [[ "${WITH_LOGS}" -eq 1 ]]; then
  mkdir -p "${LOCAL_ROOT}/logs"
  echo ""
  echo "Pulling related Condor logs..."
  scp "${REMOTE}/artifacts/logs/attribution_n100."*.{out,err,log} \
      "${LOCAL_ROOT}/logs/" 2>/dev/null || true
  scp "${REMOTE}/artifacts/logs/modality_gap."*.{out,err,log} \
      "${LOCAL_ROOT}/logs/" 2>/dev/null || true
  scp "${REMOTE}/artifacts/logs/attribution_n100-"*.log \
      "${LOCAL_ROOT}/logs/" 2>/dev/null || true
  scp "${REMOTE}/artifacts/logs/modality_gap-"*.log \
      "${LOCAL_ROOT}/logs/" 2>/dev/null || true
fi

echo ""
echo "Next (local):"
echo "  py -3 scripts/plot_modality_sensitivity.py"
