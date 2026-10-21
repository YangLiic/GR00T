#!/usr/bin/env python3
"""Build a clean-only LeRobot view: first 50 episodes (demo_clean) via symlinks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CAMS = ["head", "left", "right"]
N_CLEAN = 50


def subset_task(src: Path, dst: Path) -> None:
    info = json.loads((src / "meta" / "info.json").read_text())
    eps = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    clean = [e for e in eps if e.get("split") == "clean"]
    if not clean:
        clean = [e for e in eps if int(e["episode_index"]) < N_CLEAN]
    if len(clean) != N_CLEAN:
        raise SystemExit(f"{src.name}: expected {N_CLEAN} clean episodes, got {len(clean)}")
    clean.sort(key=lambda e: int(e["episode_index"]))
    ids = [int(e["episode_index"]) for e in clean]
    if ids != list(range(N_CLEAN)):
        raise SystemExit(f"{src.name}: clean episode ids {ids[:5]}... not 0..{N_CLEAN-1}")

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for cam in CAMS:
        (dst / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(parents=True, exist_ok=True)

    for i in range(N_CLEAN):
        pq_s = src / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
        pq_d = dst / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
        if not pq_s.exists():
            raise FileNotFoundError(pq_s)
        if pq_d.exists() or pq_d.is_symlink():
            pq_d.unlink()
        pq_d.symlink_to(os.path.relpath(pq_s, pq_d.parent))
        for cam in CAMS:
            mp_s = src / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{i:06d}.mp4"
            mp_d = dst / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{i:06d}.mp4"
            if not mp_s.exists() or mp_s.stat().st_size <= 0:
                raise FileNotFoundError(mp_s)
            if mp_d.exists() or mp_d.is_symlink():
                mp_d.unlink()
            mp_d.symlink_to(os.path.relpath(mp_s, mp_d.parent))

    meta = dst / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    used_texts = []
    seen = set()
    for e in clean:
        t = (e.get("tasks") or [None])[0]
        if t not in seen:
            seen.add(t)
            used_texts.append(t)
    src_tasks = {}
    tp = src / "meta" / "tasks.jsonl"
    if tp.exists():
        for line in tp.read_text().splitlines():
            rec = json.loads(line)
            src_tasks[rec.get("task")] = rec
    with open(meta / "episodes.jsonl", "w") as f:
        for e in clean:
            f.write(json.dumps(e) + "\n")
    with open(meta / "tasks.jsonl", "w") as f:
        for i, text in enumerate(used_texts):
            rec = src_tasks.get(text, {"task_index": i, "task": text})
            rec["task_index"] = i
            rec["task"] = text
            f.write(json.dumps(rec) + "\n")
    # Keep original task_index in parquet (maps into original tasks.jsonl).
    # Safer: copy full tasks.jsonl so parquet indices stay valid.
    (meta / "tasks.jsonl").write_text((src / "meta" / "tasks.jsonl").read_text())
    (meta / "modality.json").write_text((src / "meta" / "modality.json").read_text())
    total_frames = int(sum(int(e["length"]) for e in clean))
    info["total_episodes"] = N_CLEAN
    info["total_frames"] = total_frames
    info["total_videos"] = N_CLEAN * len(CAMS)
    info["splits"] = {"train": f"0:{N_CLEAN}"}
    info["robotwin_split"] = "clean"
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=Path, required=True)
    p.add_argument("--dst-root", type=Path, required=True)
    args = p.parse_args()
    names = sorted(
        d.name
        for d in args.src_root.iterdir()
        if d.is_dir() and (d / "meta" / "info.json").exists()
    )
    args.dst_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        subset_task(args.src_root / name, args.dst_root / name)
        print(f"ok {name}", flush=True)
    (args.dst_root / "dataset_paths.txt").write_text("\n".join(names) + "\n")
    print(f"ALL_CLEAN_SUBSET tasks={len(names)} -> {args.dst_root}", flush=True)


if __name__ == "__main__":
    main()
