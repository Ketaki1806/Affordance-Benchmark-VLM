#!/usr/bin/env bash
# Build train_1000 manifest (CPU). Prefer Condor — train JSON is large / RAM-heavy.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/cluster_paths.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/logs"

LOG_FILE="${PROJECT_ROOT}/artifacts/logs/build-train1000-$(date +%Y%m%d-%H%M%S).log"
echo "Building train_1000 manifest. Log: ${LOG_FILE}"
echo "Node: $(hostname)  Started: $(date)"

bash "${PROJECT_ROOT}/scripts/build_paco_train_1000.sh" 2>&1 | tee "${LOG_FILE}"

echo "Finished: $(date)"
ls -lh "${PROJECT_ROOT}/data/paco/manifest_train_1000.json"
