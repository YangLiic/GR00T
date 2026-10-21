#!/usr/bin/env bash
# On Licheng: wait until 50-task LeRobot convert finishes, then SSH-launch
# finetune on the borrowed A800 hold box. Do not kill hold_watchdog.sh.
set -euo pipefail

DEV_ROOT=/DRoboticsAILab/vepfs_20260622/licheng.yang
OUT="$DEV_ROOT/datasets/robotwin2.0_gr00t_lerobot"
LAUNCH=/data/efs/vepfs_20260622/licheng.yang/Isaac-GR00T/examples/RoboTwin/scripts/run_finetune_a800.sh
CRED=/DRoboticsAILab/users/lingfeng.qian/DRRM-VODPP/dag_tgcontrolA800.py
LOG="$OUT/wait_and_launch.log"

mkdir -p "$OUT"
exec >>"$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] wait_and_launch pid=$$"

need_tasks=50
deadline=$((SECONDS + 14400))
while true; do
  n=$(find "$OUT" -path '*/meta/info.json' 2>/dev/null | wc -l)
  paths=0
  if [[ -f "$OUT/dataset_paths.txt" ]]; then
    paths=$(grep -c . "$OUT/dataset_paths.txt" || true)
  fi
  echo "[$(date --iso-8601=seconds)] tasks=$n paths=$paths"
  if [[ "$n" -ge "$need_tasks" ]]; then
    python3 - "$OUT" << 'PY'
from pathlib import Path
import sys
out = Path(sys.argv[1])
names = sorted(p.parent.parent.name for p in out.glob("*/meta/info.json"))
(out / "dataset_paths.txt").write_text("\n".join(names) + "\n")
print("wrote", len(names), "paths")
PY
    break
  fi
  if [[ "$SECONDS" -gt "$deadline" ]]; then
    echo "timeout waiting for 50 tasks, have $n" >&2
    exit 1
  fi
  sleep 30
done
echo "[$(date --iso-8601=seconds)] convert complete n=$n, skip slow stats precompute"

python3 - "$CRED" << 'PY' > /tmp/a800_hold_ip.txt
import re, sys
from pathlib import Path
from volcenginesdkcore import Configuration, ApiClient
import volcenginesdkmlplatform20240701 as mlp

src = Path(sys.argv[1]).read_text()
m = re.search(
    r"def launch_two_box_pick_video\(\):.*?ak = \"(AKLT[^\"]+)\".*?sk = \"([^\"]+)\"",
    src,
    re.S,
)
if not m:
    raise SystemExit("creds not found")
cfg = Configuration()
cfg.ak, cfg.sk, cfg.region = m.group(1), m.group(2), "cn-beijing"
api = mlp.MLPLATFORM20240701Api(ApiClient(cfg))
jobs = api.list_jobs(mlp.ListJobsRequest(
    name_contains="vodpp-hold-lfqian",
    page_number=1, page_size=5,
    sort_by="CreateTime", sort_order="Descend",
))
job = jobs.items[0]
inst = api.list_job_instances(mlp.ListJobInstancesRequest(job_id=job.id, page_number=1, page_size=5))
ip = inst.items[0].ips.primary_ip
state = inst.items[0].status.state
print(ip)
print(state, file=sys.stderr)
print(job.id, file=sys.stderr)
PY
IP=$(head -1 /tmp/a800_hold_ip.txt)
echo "A800_IP=$IP"
rm -f /tmp/a800_hold_ip.txt

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p 12223 "root@$IP" \
  "nohup setsid bash $LAUNCH </dev/null >/tmp/run_gr00t_n17_robotwin.nohup 2>&1 & echo launched_pid=\$!"
echo "[$(date --iso-8601=seconds)] ssh launch issued"
