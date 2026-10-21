#!/usr/bin/env python3
"""Prepare 4 AR5 LeRobot datasets for GR00T N1.7 finetune.

Writes task prompts, 16-D 7+1+7+1 modality slices, annotation metadata,
and GR00T ``meta/stats.json``. Does not start training.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_ROOT = Path(
    os.environ.get(
        "AR5_DATA_ROOT",
        "/DRoboticsAILab/users/licheng.yang/datasets/D-Robotics",
    )
)

TASKS = {
    "assembly_workpiece_more_ar5": "assemble the workpiece",
    "stack_and_cover_480_ar5": "stack two cubes and cover them with a cup",
    "pick_and_place_ar5": "pick up three cubes and place them in the corresponding color regions",
    "drag_drawer_and_pick_place_ar5": "open the drawer and put the cube inside",
}

JOINT_SLICES = {
    "left_arm": {"start": 0, "end": 7},
    "left_gripper": {"start": 7, "end": 8},
    "right_arm": {"start": 8, "end": 15},
    "right_gripper": {"start": 15, "end": 16},
}

STATS_FINGERPRINTS_KEY = "__fingerprints__"
LE_ROBOT_DATA_GLOB = "data/*/*.parquet"


def _backup_once(path: Path) -> None:
    bak = path.with_name(path.name + ".bak")
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _update_modality(meta_dir: Path) -> None:
    path = meta_dir / "modality.json"
    _backup_once(path)
    modality = json.loads(path.read_text()) if path.exists() else {}
    modality["state"] = {k: dict(v) for k, v in JOINT_SLICES.items()}
    modality["action"] = {k: dict(v) for k, v in JOINT_SLICES.items()}
    modality["annotation"] = {
        "human.task_description": {"original_key": "task_index"}
    }
    _write_json(path, modality)


def _update_prompts(meta_dir: Path, prompt: str) -> None:
    tasks_path = meta_dir / "tasks.jsonl"
    _backup_once(tasks_path)
    tasks_path.write_text(json.dumps({"task_index": 0, "task": prompt}, ensure_ascii=False) + "\n")

    episodes_path = meta_dir / "episodes.jsonl"
    _backup_once(episodes_path)
    rows = []
    for line in episodes_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["tasks"] = [prompt]
        rows.append(row)
    episodes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def _fingerprint(feature_name: str, feature_meta: dict) -> str:
    payload = {
        "feature": feature_name,
        "dtype": feature_meta.get("dtype"),
        "shape": feature_meta.get("shape"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _generate_stats(dataset_path: Path) -> None:
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    features = info["features"]
    lowdim = [name for name, meta in features.items() if "float" in meta.get("dtype", "")]
    parquet_paths = sorted(dataset_path.glob(LE_ROBOT_DATA_GLOB))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet under {dataset_path}")

    frames = []
    for parquet_path in tqdm(parquet_paths, desc=f"stats {dataset_path.name}"):
        frames.append(pd.read_parquet(parquet_path, columns=lowdim))
    data = pd.concat(frames, axis=0, ignore_index=True)

    stats: dict = {}
    fingerprints: dict = {}
    for name in lowdim:
        np_data = np.vstack([np.asarray(x, dtype=np.float32) for x in data[name]])
        stats[name] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
        fingerprints[name] = _fingerprint(name, features[name])
    stats[STATS_FINGERPRINTS_KEY] = fingerprints
    out = dataset_path / "meta" / "stats.json"
    out.write_text(json.dumps(stats, indent=4) + "\n")


def _verify(dataset_path: Path, prompt: str) -> dict:
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    tasks = [json.loads(l) for l in (dataset_path / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()]
    episodes = [json.loads(l) for l in (dataset_path / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    modality = json.loads((dataset_path / "meta" / "modality.json").read_text())
    lengths = [int(e["length"]) for e in episodes]
    parquet_files = sorted((dataset_path / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet under {dataset_path}")
    pq0 = parquet_files[0]
    df = pd.read_parquet(pq0, columns=["observation.state", "action"])
    state = np.stack(df["observation.state"].to_numpy())
    action = np.stack(df["action"].to_numpy())
    cams = {}
    video_root = dataset_path / "videos" / "chunk-000"
    for cam in ("observation.images.head_camera", "observation.images.left_wrist_camera", "observation.images.right_wrist_camera"):
        cams[cam] = len(list((video_root / cam).glob("*.mp4"))) if (video_root / cam).is_dir() else 0
    image_shapes = {
        k: v.get("shape")
        for k, v in info["features"].items()
        if k.startswith("observation.images")
    }
    return {
        "name": dataset_path.name,
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "fps": info["fps"],
        "prompt": tasks[0]["task"] if tasks else "",
        "prompt_ok": tasks[0]["task"] == prompt and all(e.get("tasks") == [prompt] for e in episodes),
        "len_min": min(lengths),
        "len_max": max(lengths),
        "n_lt_45": sum(l < 45 for l in lengths),
        "state_dim": int(state.shape[-1]),
        "action_dim": int(action.shape[-1]),
        "state_eq_action": bool(np.allclose(state, action)),
        "gripper_state_minmax": [
            [float(state[:, 7].min()), float(state[:, 7].max())],
            [float(state[:, 15].min()), float(state[:, 15].max())],
        ],
        "gripper_action_minmax": [
            [float(action[:, 7].min()), float(action[:, 7].max())],
            [float(action[:, 15].min()), float(action[:, 15].max())],
        ],
        "image_shapes": image_shapes,
        "n_videos": cams,
        "has_stats": (dataset_path / "meta" / "stats.json").exists(),
        "state_keys": list(modality["state"]),
        "annotation": modality.get("annotation"),
    }


def main() -> None:
    reports = []
    for name, prompt in TASKS.items():
        dataset_path = DATA_ROOT / name
        if not (dataset_path / "meta" / "info.json").exists():
            raise FileNotFoundError(dataset_path)
        meta_dir = dataset_path / "meta"
        print(f"==> prepare {dataset_path}")
        _update_modality(meta_dir)
        _update_prompts(meta_dir, prompt)
        _generate_stats(dataset_path)
        reports.append(_verify(dataset_path, prompt))
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    bad = [r["name"] for r in reports if not r["prompt_ok"] or r["n_lt_45"] or r["state_dim"] != 16]
    if bad:
        raise SystemExit(f"prepare failed: {bad}")
    print("AR5_PREPARE_OK")


if __name__ == "__main__":
    main()
