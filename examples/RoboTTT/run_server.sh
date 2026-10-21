#!/usr/bin/env bash
set -euo pipefail

TASK_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_REPO_ROOT"
# Sharing the original venv is safe only when imports resolve to this checkout.
export PYTHONPATH="$TASK_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${MODEL_PATH:?Set a trained RoboTTT checkpoint}"
: "${EMBODIMENT_TAG:?Set the checkpoint embodiment tag}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# For live SAPIEN RGB use ROBOTWIN_BGR2RGB=0. Reset the policy at episode boundaries.
exec "$PYTHON_BIN" gr00t/eval/run_gr00t_server.py \
  --model-path "$MODEL_PATH" --embodiment-tag "$EMBODIMENT_TAG" "$@"
