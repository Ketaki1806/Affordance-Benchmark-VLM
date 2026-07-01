#!/bin/bash
#SBATCH --job-name=install-deps
#SBATCH --output=artifacts/logs/install-deps-%j.out
#SBATCH --error=artifacts/logs/install-deps-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# Install pip GPU packages on a compute node (avoids login-node OOM kills).
# Usage: sbatch scripts/install_deps.slurm

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${PROJECT_ROOT}/artifacts/logs"

bash "${PROJECT_ROOT}/scripts/install_deps.sh"
