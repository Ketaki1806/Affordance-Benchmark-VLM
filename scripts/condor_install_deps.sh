#!/usr/bin/env bash
# HTCondor wrapper: install GPU deps on an execution node.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
bash "${PROJECT_ROOT}/scripts/install_deps.sh"
