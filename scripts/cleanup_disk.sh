#!/usr/bin/env bash
# Free disk space in home directory after failed installs.
# Usage: bash scripts/cleanup_disk.sh

set -euo pipefail

echo "=== Disk usage before ==="
du -sh ~ 2>/dev/null || true
du -sh ~/.cache ~/micromamba ~/miniconda3 ~/miniforge3 2>/dev/null || true

echo ""
echo "Cleaning package caches..."
rm -rf ~/.cache/pip ~/.cache/conda ~/.cache/huggingface 2>/dev/null || true
micromamba clean --all -y 2>/dev/null || true

echo ""
echo "Removing failed Miniconda install (if present)..."
rm -rf ~/miniconda3

echo ""
echo "=== Disk usage after ==="
du -sh ~ 2>/dev/null || true
du -sh ~/.cache ~/micromamba 2>/dev/null || true

echo ""
echo "If still full, move env to scratch:"
echo "  source scripts/cluster_paths.sh"
echo "  bash scripts/setup_env_micromamba.sh"
