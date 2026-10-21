#!/usr/bin/env bash
set -euo pipefail
export SHARD=machine4 ENCODER=h264_nvenc NGPU=8 WORKERS=24 CPU_WORKERS=48
exec bash /DRoboticsAILab/vepfs_20260622/licheng.yang/Isaac-GR00T/examples/RoboTwin/scripts/convert_shard.sh \
  stack_blocks_three stack_blocks_two stack_bowls_three \
  stack_bowls_two stamp_seal turn_switch
