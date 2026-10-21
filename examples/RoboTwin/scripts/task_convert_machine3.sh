#!/usr/bin/env bash
set -euo pipefail
export SHARD=machine3 ENCODER=h264_nvenc NGPU=8 WORKERS=24 CPU_WORKERS=48
exec bash /DRoboticsAILab/vepfs_20260622/licheng.yang/Isaac-GR00T/examples/RoboTwin/scripts/convert_shard.sh \
  put_bottles_dustbin put_object_cabinet rotate_qrcode \
  scan_object shake_bottle shake_bottle_horizontally
