#!/usr/bin/env bash
# HTCondor wrapper: run one caption-pipeline shard on a GPU node.
# Args: <shard_index> <num_shards>
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p artifacts/logs

SHARD_INDEX="${1:?shard_index required}"
NUM_SHARDS="${2:?num_shards required}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/activate_env.sh"

echo "Job ID:      ${CONDOR_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Shard:       ${SHARD_INDEX} / ${NUM_SHARDS}"
echo "Started at:  $(date)"

nvidia-smi || true
bash "${PROJECT_ROOT}/scripts/run_pipeline.sh" \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}"

echo "Finished at: $(date)"
