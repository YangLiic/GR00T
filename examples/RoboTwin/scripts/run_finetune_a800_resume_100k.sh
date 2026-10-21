#!/usr/bin/env bash
# Resume RoboTwin clean-only GR00T N1.7 finetune from checkpoint-40000 to 100k steps.
# Uses --resume-from-checkpoint (optimizer + DeepSpeed states + step counter).
# Drops scheduler.pt on the latest checkpoint so cosine LR is rebuilt for a
# 100k-step schedule instead of staying at the 40k floor (~0).
set -euo pipefail

DEV_ROOT=/DRoboticsAILab/vepfs_20260622/licheng.yang
A800_ROOT=/data/efs/vepfs_20260622/licheng.yang
if [[ -d "$A800_ROOT/Isaac-GR00T" ]]; then
  ROOT="$A800_ROOT"
  CKPT_ROOT=/data/efs/user/licheng.yang/checkpoints/gr00t
else
  ROOT="$DEV_ROOT"
  CKPT_ROOT=/DRoboticsAILab/users/licheng.yang/checkpoints/gr00t
fi

GR00T="$ROOT/Isaac-GR00T"
DATA="$ROOT/datasets/robotwin2.0_gr00t_lerobot_clean"
SITE="$GR00T/.venv/lib/python3.12/site-packages"
PY_PREFIX=""
for cand in \
  "$ROOT/.uv_share/python/cpython-3.12.2-linux-x86_64-gnu" \
  "$ROOT/.uv_share/python/cpython-3.12.12-linux-x86_64-gnu"
do
  if [[ -x "$cand/bin/python3.12" ]]; then
    PY_PREFIX="$cand"
    break
  fi
done
UVPY="${PY_PREFIX:+$PY_PREFIX/bin/python3.12}"
LOG_DIR="$CKPT_ROOT/n17_robotwin50_clean_joint_h32"
RUN_DIR="$LOG_DIR/n17_robotwin50_clean_joint_h32"
LOG="$LOG_DIR/train_resume_100k.log"
PATHS_FILE="$DATA/dataset_paths.txt"
MODALITY="$GR00T/examples/RoboTwin/robotwin_aloha_config.py"
BASE="$GR00T/checkpoints/GR00T-N1.7-3B"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date --iso-8601=seconds)] resume wrapper pid=$$ host=$(hostname) root=$ROOT"

if [[ -z "$UVPY" || ! -x "$UVPY" ]]; then
  echo "missing portable python under $ROOT/.uv_share/python" >&2
  exit 1
fi

LATEST_CKPT="$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
[[ -n "$LATEST_CKPT" && -d "$LATEST_CKPT" ]] || {
  echo "no checkpoint-* under $RUN_DIR" >&2
  exit 1
}
echo "latest_checkpoint=$LATEST_CKPT"
if [[ -f "$LATEST_CKPT/scheduler.pt" ]]; then
  mv -f "$LATEST_CKPT/scheduler.pt" "$LATEST_CKPT/scheduler.pt.bak_40k"
  echo "moved scheduler.pt -> scheduler.pt.bak_40k (rebuild cosine for 100k total steps)"
fi

if [[ ! -f "$PATHS_FILE" ]]; then
  echo "missing $PATHS_FILE" >&2
  exit 1
fi

if [[ -x /tmp/stop_hold.sh ]]; then
  echo $$ > /tmp/drrm_hold_work_active
  bash /tmp/stop_hold.sh || true
  for i in $(seq 1 30); do
    n=$(pgrep -cf "pretrain_loade[r].py" || true)
    [[ "${n:-0}" -eq 0 ]] && break
    sleep 2
  done
fi

env -u PYTHONHOME -u PYTHONPATH bash "$GR00T/examples/RoboTwin/scripts/link_local_hf_checkpoints.sh" "$GR00T"

mapfile -t TASKS < "$PATHS_FILE"
DATASET_PATHS=""
for t in "${TASKS[@]}"; do
  [[ -z "$t" ]] && continue
  d="$DATA/$t"
  [[ -f "$d/meta/info.json" ]] || { echo "missing dataset $d/meta/info.json" >&2; exit 1; }
  if [[ -z "$DATASET_PATHS" ]]; then
    DATASET_PATHS="$d"
  else
    DATASET_PATHS="$DATASET_PATHS:$d"
  fi
done
echo "n_datasets=$(grep -c . "$PATHS_FILE")"

export PYTHONHOME="$PY_PREFIX"
export LD_LIBRARY_PATH="$PY_PREFIX/lib:${SITE}/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$GR00T:$SITE"
export PYTHONNOUSERSITE=1
export HF_HOME="$GR00T/.hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DIR="$LOG_DIR"
export WANDB_CACHE_DIR="$LOG_DIR/wandb_cache"
export WANDB_PROJECT=finetune-gr00t-n1d7
export WANDB_ANONYMOUS=allow
export NO_ALBUMENTATIONS_UPDATE=1
export SETUPTOOLS_USE_DISTUTILS=local
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$GR00T"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "NUM_GPUS=$NUM_GPUS resume_from_checkpoint=True max_steps=100000"

cleanup() { rm -f /tmp/drrm_hold_work_active; }
trap cleanup EXIT INT TERM HUP

"$UVPY" -c "import gr00t, torch; print('gr00t_ok torch', torch.__version__, 'cuda', torch.cuda.is_available())"

"$UVPY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port=29501 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE" \
  --dataset-path "$DATASET_PATHS" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$LOG_DIR" \
  --experiment-name n17_robotwin50_clean_joint_h32 \
  --global-batch-size 64 \
  --max-steps 100000 \
  --save-steps 5000 \
  --save-total-limit 5 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --dataloader-num-workers 4 \
  --state-dropout-prob 0.2 \
  --ds-weights-alpha 0.0 \
  --resume-from-checkpoint \
  --tune-visual \
  --use-wandb \
  --wandb-project finetune-gr00t-n1d7 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
