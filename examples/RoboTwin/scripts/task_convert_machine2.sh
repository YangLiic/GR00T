#!/usr/bin/env bash
set -euo pipefail
export SHARD=machine2 ENCODER=h264_nvenc NGPU=8 WORKERS=24 CPU_WORKERS=48
exec bash /DRoboticsAILab/vepfs_20260622/licheng.yang/Isaac-GR00T/examples/RoboTwin/scripts/convert_shard.sh \
  place_object_basket place_object_scale place_object_stand \
  place_phone_stand place_shoe press_stapler
