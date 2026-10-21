#!/usr/bin/env bash
# GR00T N1.7 co-training on 3 AR5 real-robot tasks, submitted via gpu_slot_ctl
# onto licheng.yang's own 8xA800 job (t-20260907191950-fjv5z).
# Absolute 16-D joints, action horizon 40, supervise the LeRobot action column
# (no --copy-state-to-action). Prompts come from each dataset's tasks.jsonl.
set -euo pipefail

ROOT=/DRoboticsAILab/vepfs_20260622/licheng.yang
CKPT_ROOT=/DRoboticsAILab/users/licheng.yang/checkpoints/gr00t
DATA_ROOT=/DRoboticsAILab/users/licheng.yang/datasets/D-Robotics

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
EXP_NAME=n17_ar5_3task_joint_h40
LOG_DIR="$CKPT_ROOT/$EXP_NAME"
LOG="$LOG_DIR/train.log"
MODALITY="$GR00T/examples/AR5/ar5_config.py"
BASE="$GR00T/checkpoints/GR00T-N1.7-3B"

TASKS=(
  stack_and_cover_0905_ar5
  drag_the_drawer_0901_ar5
  assembly_workpiece_0902_ar5
)

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date --iso-8601=seconds)] wrapper pid=$$ host=$(hostname) root=$ROOT data=$DATA_ROOT"

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
echo "NUM_GPUS=$NUM_GPUS max_steps=100000 copy_state_to_action=0"

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
  --max-steps 100000 \
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
