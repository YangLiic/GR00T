#!/usr/bin/env bash
set -euo pipefail

# No task or GPU allocation is selected implicitly.
TASK_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_REPO_ROOT"
export PYTHONPATH="$TASK_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_MODEL_PATH:?Set an existing local GR00T checkpoint}"
: "${DATASET_PATH:?Set a LeRobot task dataset}"
: "${OUTPUT_DIR:?Set a new output directory}"
: "${MODALITY_CONFIG_PATH:?Set the task/robot modality Python config}"
: "${EMBODIMENT_TAG:?Set the matching embodiment tag}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Set ROBOTWIN_BGR2RGB explicitly according to the dataset; no generic color guess.
if [[ -e "$OUTPUT_DIR/config.json" || -d "$OUTPUT_DIR/checkpoint-1000" ]]; then
  echo "Choose a fresh output directory, or use the explicit resume CLI." >&2
  exit 1
fi

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL_PATH" --dataset-path "$DATASET_PATH" \
  --embodiment-tag "$EMBODIMENT_TAG" --modality-config-path "$MODALITY_CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" --num-gpus "$NUM_GPUS" --global-batch-size "$NUM_GPUS" \
  --robottt-enabled --robottt-train-mode posttrain \
  --robottt-context-length "${CONTEXT_LENGTH:-1024}" \
  --robottt-tbptt-steps "${TBPTT_STEPS:-4}" \
  --learning-rate 5e-5 --weight-decay 1e-5 --max-steps "${MAX_STEPS:-20000}" \
  --save-steps 1000 --dataloader-num-workers 0 \
  --tune-llm --tune-visual --tune-projector --tune-diffusion-model "$@"
