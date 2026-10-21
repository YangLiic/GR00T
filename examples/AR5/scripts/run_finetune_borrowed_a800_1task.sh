#!/usr/bin/env bash
# GR00T N1.7 单任务微调（每台机器一个 AR5 任务），与 openpi 那三条一一对应。
# 16 维绝对关节、action chunk 40、监督 LeRobot 的 action 列（不加 --copy-state-to-action）。
# prompt 取该库自己的 tasks.jsonl。任务靠环境变量 AR5_TASK 指定。
#
# 本文件是「借同事 A800」专用版：A800 上没有 /DRoboticsAILab，路径全走 /data/efs/...。
# 并按借卡红线收尾：写 marker -> stop_hold -> 训练 -> trap 清 marker 让 watchdog 自动还卡。
# 绝不 kill hold_watchdog.sh。
set -euo pipefail

ROOT=/data/efs/vepfs_20260622/licheng.yang
CKPT_ROOT=/data/efs/user/licheng.yang/checkpoints/gr00t
DATA_ROOT=/data/efs/user/licheng.yang/datasets/D-Robotics

GR00T="$ROOT/Isaac-GR00T"
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
AR5_TASK="${AR5_TASK:?必须设置 AR5_TASK（数据集目录名）}"
MAX_STEPS="${MAX_STEPS:-40000}"
EXP_NAME="${EXP_NAME:-n17_ar5_${AR5_TASK}_act_h40}"
LOG_DIR="$CKPT_ROOT/$EXP_NAME"
LOG="$LOG_DIR/train.log"
MODALITY="$GR00T/examples/AR5/ar5_config.py"
BASE="$GR00T/checkpoints/GR00T-N1.7-3B"

TASKS=( "$AR5_TASK" )

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date --iso-8601=seconds)] wrapper pid=$$ host=$(hostname) root=$ROOT data=$DATA_ROOT"

# ---- 借卡交接（红线：只停 pretrain_loader.py，绝不碰 hold_watchdog.sh）----
echo $$ > /tmp/drrm_hold_work_active
echo "[hold] work marker -> $$"
cleanup() { rm -f /tmp/drrm_hold_work_active; }
trap cleanup EXIT INT TERM HUP
bash /tmp/stop_hold.sh || true
for i in $(seq 1 30); do
  n=$(pgrep -cf "pretrain_loade[r].py" || true)
  echo "[hold] loader processes=${n:-0}"
  [[ "${n:-0}" -eq 0 ]] && break
  sleep 2
done
# --------------------------------------------------------------------------

if [[ -z "$UVPY" || ! -x "$UVPY" ]]; then
  echo "missing portable python under $ROOT/.uv_share/python" >&2
  exit 1
fi
echo "python=$UVPY py_prefix=$PY_PREFIX"

DATASET_PATHS=""
for t in "${TASKS[@]}"; do
  d="$DATA_ROOT/$t"
  [[ -f "$d/meta/info.json" ]] || { echo "missing dataset $d/meta/info.json" >&2; exit 1; }
  [[ -f "$d/meta/stats.json" ]] || { echo "missing $d/meta/stats.json ; run prepare_ar5_3task_lerobot.py first" >&2; exit 1; }
  [[ -f "$d/meta/tasks.jsonl" ]] || { echo "missing $d/meta/tasks.jsonl" >&2; exit 1; }
  if [[ -z "$DATASET_PATHS" ]]; then
    DATASET_PATHS="$d"
  else
    DATASET_PATHS="$DATASET_PATHS:$d"
  fi
  echo "prompt $t: $(python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["task"])' "$d/meta/tasks.jsonl")"
done
echo "n_datasets=${#TASKS[@]} dataset_path_chars=${#DATASET_PATHS}"

env -u PYTHONHOME -u PYTHONPATH bash "$GR00T/examples/RoboTwin/scripts/link_local_hf_checkpoints.sh" "$GR00T"

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
echo "NUM_GPUS=$NUM_GPUS task=$AR5_TASK max_steps=$MAX_STEPS copy_state_to_action=0（监督 action 列）"

"$UVPY" -c "import gr00t, torch; print('gr00t_ok torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"

"$UVPY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port=29501 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE" \
  --dataset-path "$DATASET_PATHS" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$LOG_DIR" \
  --experiment-name "$EXP_NAME" \
  --global-batch-size 64 \
  --max-steps "$MAX_STEPS" \
  --save-steps 5000 \
  --save-total-limit 5 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --dataloader-num-workers 4 \
  --state-dropout-prob 0.2 \
  --ds-weights-alpha 0.0 \
  --tune-visual \
  --use-wandb \
  --wandb-project finetune-gr00t-n1d7 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
