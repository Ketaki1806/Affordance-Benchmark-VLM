#!/bin/bash
#SBATCH --job-name=affordance-caption
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# Adjust partition/account for your cluster, for example:
# #SBATCH --partition=gpu
# #SBATCH --account=your_project_account

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="affordance_benchmark"

cd "${PROJECT_ROOT}"
mkdir -p artifacts/logs

echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:     ${PROJECT_ROOT}"
echo "Started at:  $(date)"

# Load cluster modules if needed (uncomment and edit for your site):
# module purge
# module load cuda/12.1
# module load miniconda3

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
else
  echo "ERROR: conda not found on compute node."
  exit 1
fi

nvidia-smi || true

bash "${PROJECT_ROOT}/scripts/run_pipeline.sh"

echo "Finished at: $(date)"
