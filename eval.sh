#!/usr/bin/env bash
# Evaluate a released PRISM actor on a published box or ball scene.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
cd "$repo_root"
exec "${PRISM_PYTHON:-python3}" -u "$repo_root/scripts/run_eval.py" "$@"
