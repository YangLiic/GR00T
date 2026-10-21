#!/usr/bin/env bash
# Point HuggingFace cache at the local Cosmos-Reason2-2B snapshot so N1.7
# can load the gated backbone offline.
set -euo pipefail

ROOT="${1:-/data/efs/vepfs_20260622/licheng.yang/Isaac-GR00T}"
HF_HOME="${HF_HOME:-$ROOT/.hf_home}"
COSMOS="$ROOT/checkpoints/Cosmos-Reason2-2B"
REV="$(cat "$HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B/refs/main" 2>/dev/null || true)"
REV="${REV:-9ce19a195e423419c349abfc86fd07178b230561}"

SNAP="$HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/$REV"
mkdir -p "$SNAP" "$HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B/refs"
printf '%s' "$REV" > "$HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B/refs/main"

if [[ ! -f "$COSMOS/model.safetensors" ]]; then
  echo "missing local cosmos weights: $COSMOS/model.safetensors" >&2
  exit 1
fi

# Relative links so the same EFS tree works on Licheng (/DRoboticsAILab/...)
# and the A800 box (/data/efs/...). Absolute /DRoboticsAILab paths are broken there.
rel="$(env -u PYTHONHOME -u PYTHONPATH python3 -c "import os; print(os.path.relpath('$COSMOS', '$SNAP'))")"
shopt -s nullglob
for f in "$COSMOS"/*; do
  base="$(basename "$f")"
  [[ "$base" == ".cache" ]] && continue
  ln -sfn "$rel/$base" "$SNAP/$base"
done

if [[ ! -e "$SNAP/config.json" ]]; then
  echo "broken cosmos snapshot link: $SNAP/config.json" >&2
  ls -l "$SNAP/config.json" >&2 || true
  exit 1
fi

# Local path used by launch_finetune.py (string still contains nvidia/Cosmos-Reason2).
LOCAL="$ROOT/checkpoints/nvidia/Cosmos-Reason2-2B"
mkdir -p "$ROOT/checkpoints/nvidia"
ln -sfn "../Cosmos-Reason2-2B" "$LOCAL"
if [[ ! -e "$LOCAL/config.json" ]]; then
  echo "broken local cosmos path: $LOCAL" >&2
  exit 1
fi

echo "linked cosmos snapshot -> $SNAP (rel=$rel)"
echo "local cosmos path -> $LOCAL"
ls -l "$SNAP/config.json" "$LOCAL/config.json" "$SNAP/model.safetensors"
