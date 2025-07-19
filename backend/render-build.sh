#!/usr/bin/env bash
echo "=== Installing Git LFS and pulling large files ==="
apt-get update && apt-get install -y git-lfs
git lfs install
git lfs pull
