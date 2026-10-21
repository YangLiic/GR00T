#!/usr/bin/env bash
# Run a disjoint task shard of HDF5 -> LeRobot. Works on Licheng and L4 slots.
set -euo pipefail
ROOT=/DRoboticsAILab/vepfs_20260622/licheng.yang
GR00T="$ROOT/Isaac-GR00T"
OUT="$ROOT/datasets/robotwin2.0_gr00t_lerobot"
PY_PREFIX="$ROOT/.uv_share/python/cpython-3.12.2-linux-x86_64-gnu"
SITE="$GR00T/.venv/lib/python3.12/site-packages"
FFMPEG_DIR="$ROOT/.uv_share/ffmpeg"
SHARD="${SHARD:-local}"
WORKERS="${WORKERS:-48}"
ENCODER="${ENCODER:-libx264}"
NGPU="${NGPU:-0}"
LOG="$OUT/convert_${SHARD}.log"

if [[ $# -lt 1 ]]; then
  echo "usage: convert_shard.sh <task> [task...]" >&2
  exit 1
fi

mkdir -p "$OUT"
export PATH="$FFMPEG_DIR/bin:/usr/bin:/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_DIR/lib:${PY_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONHOME="$PY_PREFIX"
export PYTHONPATH="$GR00T:$SITE"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg missing" >&2
  exit 1
fi
if [[ ! -x "$PY_PREFIX/bin/python3.12" ]]; then
  echo "missing python $PY_PREFIX" >&2
  exit 1
fi

if [[ "$ENCODER" == "h264_nvenc" ]]; then
  if ! ffmpeg -y -loglevel error -f lavfi -i color=c=black:s=320x240:d=0.2 \
      -c:v h264_nvenc -preset p1 -gpu 0 -f null - ; then
    echo "[warn] nvenc smoke failed; falling back to libx264" | tee -a "$LOG"
    ENCODER=libx264
    NGPU=0
    WORKERS="${CPU_WORKERS:-48}"
  fi
fi

echo "[$(date --iso-8601=seconds)] shard=$SHARD host=$(hostname) encoder=$ENCODER workers=$WORKERS ngpu=$NGPU tasks=$*" | tee -a "$LOG"
exec "$PY_PREFIX/bin/python3.12" "$GR00T/examples/RoboTwin/scripts/convert_robotwin_to_lerobot.py" \
  --official-root "$ROOT/datasets/robotwin2.0_official" \
  --out-root "$OUT" \
  --workers "$WORKERS" \
  --encoder "$ENCODER" \
  --ngpu "$NGPU" \
  --tasks "$@" >>"$LOG" 2>&1
