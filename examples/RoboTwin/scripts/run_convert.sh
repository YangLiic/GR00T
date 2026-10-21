#!/usr/bin/env bash
# HDF5 (clean+randomized) -> GR00T LeRobot. Safe to resume.
set -euo pipefail
ROOT="${ROOT:-/DRoboticsAILab/vepfs_20260622/licheng.yang}"
GR00T="$ROOT/Isaac-GR00T"
OUT="$ROOT/datasets/robotwin2.0_gr00t_lerobot"
PY="$GR00T/.venv/bin/python"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$GR00T"
exec "$PY" "$GR00T/examples/RoboTwin/scripts/convert_robotwin_to_lerobot.py" \
  --official-root "$ROOT/datasets/robotwin2.0_official" \
  --out-root "$OUT" \
  --workers "${WORKERS:-32}" \
  "$@"
