#!/usr/bin/env bash
set -euo pipefail
export SHARD=machine1 ENCODER=h264_nvenc NGPU=8 WORKERS=24 CPU_WORKERS=48
exec bash /DRoboticsAILab/vepfs_20260622/licheng.yang/Isaac-GR00T/examples/RoboTwin/scripts/convert_shard.sh \
  place_cans_plasticbox place_container_plate place_dual_shoes \
  place_empty_cup place_fan place_mouse_pad
