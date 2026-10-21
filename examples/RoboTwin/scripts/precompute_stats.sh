#!/usr/bin/env bash
# Precompute GR00T stats for each converted task so rank0 is not blocked at train start.
set -euo pipefail
ROOT="${ROOT:-/DRoboticsAILab/vepfs_20260622/licheng.yang}"
GR00T="$ROOT/Isaac-GR00T"
DATA="$ROOT/datasets/robotwin2.0_gr00t_lerobot"
PY="$GR00T/.venv/bin/python"
MODALITY="$GR00T/examples/RoboTwin/robotwin_aloha_config.py"
LOG="$DATA/precompute_stats.log"
mkdir -p "$DATA"
exec >>"$LOG" 2>&1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$GR00T"
cd "$GR00T"
echo "[$(date --iso-8601=seconds)] precompute_stats pid=$$"

while true; do
  did=0
  for info in "$DATA"/*/meta/info.json; do
    [[ -f "$info" ]] || continue
    task_dir="$(dirname "$(dirname "$info")")"
    if [[ -f "$task_dir/meta/stats.json" && -f "$task_dir/meta/relative_stats.json" ]]; then
      continue
    fi
    echo "[$(date --iso-8601=seconds)] stats $task_dir"
    "$PY" -m gr00t.data.stats \
      --dataset-path "$task_dir" \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path "$MODALITY"
    did=1
  done
  convert_alive=0
  if [[ -f "$DATA/convert.pid" ]] && kill -0 "$(cat "$DATA/convert.pid")" 2>/dev/null; then
    convert_alive=1
  fi
  n=$(find "$DATA" -path '*/meta/info.json' 2>/dev/null | wc -l)
  s=$(find "$DATA" -path '*/meta/stats.json' 2>/dev/null | wc -l)
  echo "[$(date --iso-8601=seconds)] info=$n stats=$s convert_alive=$convert_alive"
  if [[ "$convert_alive" -eq 0 && "$s" -ge 50 ]]; then
    echo "ALL_STATS_DONE"
    exit 0
  fi
  if [[ "$convert_alive" -eq 0 && "$n" -lt 50 && "$did" -eq 0 ]]; then
    echo "convert gone, stats incomplete info=$n stats=$s" >&2
    exit 1
  fi
  sleep 30
done
