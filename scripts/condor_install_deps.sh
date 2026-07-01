#!/usr/bin/env bash
# HTCondor wrapper: install GPU deps on an execution node.
# Runs from nethome (should_transfer_files = NO in submit file).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "HOME:         ${HOME}"
echo "Node:         $(hostname)"

bash "${PROJECT_ROOT}/scripts/install_deps.sh"
