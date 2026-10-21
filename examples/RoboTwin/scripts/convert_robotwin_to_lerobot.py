#!/usr/bin/env python3
"""Convert RoboTwin 2.0 official HDF5 (clean + randomized) to GR00T LeRobot v2.

Each task becomes one LeRobot dataset mixing clean_50 and randomized_500.
Cameras: head / left / right. EEF stored as xyz+rot6d (N1.7 XYZ_ROT6D).
RoboTwin endpose quaternion is transforms3d/SAPIEN wxyz.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


FPS = 30
CAMERAS = {
    "head": "head_camera",
    "left": "left_camera",
    "right": "right_camera",
}
STATE_DIM = 32
ACTION_DIM = 20
VIDEO_KEYS = [f"observation.images.{name}" for name in CAMERAS]


def eef_xyz_wxyz_to_rot6d(pose7: np.ndarray) -> np.ndarray:
    """pose7 = xyz + quat_wxyz -> xyz + rot6d (first two rows of R)."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    xyz = pose7[..., :3]
    wxyz = pose7[..., 3:7]
    xyzw = np.concatenate([wxyz[..., 1:4], wxyz[..., 0:1]], axis=-1)
    mats = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(pose7.shape[:-1] + (3, 3))
    rot6d = mats[..., :2, :].reshape(pose7.shape[:-1] + (6,))
    return np.concatenate([xyz, rot6d], axis=-1).astype(np.float32)


def jpeg_to_bgr(raw) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    data = bytes(raw).rstrip(b"\x00")
    eoi = data.rfind(b"\xff\xd9")
    if eoi >= 0:
        data = data[: eoi + 2]
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode jpeg")
    return img


def write_mp4(
    path: Path,
    frames_bgr: list[np.ndarray],
    fps: int = FPS,
    encoder: str = "libx264",
    gpu_id: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
    ]
    if encoder == "h264_nvenc":
        cmd += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-rc",
            "constqp",
            "-qp",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]
        if gpu_id is not None:
            cmd += ["-gpu", str(gpu_id)]
    else:
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "1",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]
    cmd.append(str(path))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames_bgr:
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed rc={rc} path={path}")


def load_instruction(instr_path: Path, fallback: str) -> str:
    if not instr_path.exists():
        return fallback
    data = json.loads(instr_path.read_text())
    if isinstance(data, dict):
        seen = data.get("seen") or data.get("unseen") or []
        if seen:
            return str(seen[0])
    return fallback


def convert_episode(job: dict) -> dict:
    src = Path(job["hdf5"])
    out_root = Path(job["out_root"])
    ep_idx = int(job["ep_idx"])
    chunk = ep_idx // 1000
    fps = int(job["fps"])
    instruction = job["instruction"]

    parquet = out_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
    videos = {
        name: out_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"observation.images.{name}"
        / f"episode_{ep_idx:06d}.mp4"
        for name in CAMERAS
    }
    if parquet.exists() and all(p.exists() and p.stat().st_size > 0 for p in videos.values()):
        length = int(pd.read_parquet(parquet, columns=["index"]).shape[0])
        return {
            "ep_idx": ep_idx,
            "length": length,
            "instruction": instruction,
            "skipped": True,
            "hw": job.get("hw"),
        }

    with h5py.File(src, "r") as f:
        left_eef = eef_xyz_wxyz_to_rot6d(f["endpose/left_endpose"][:])
        right_eef = eef_xyz_wxyz_to_rot6d(f["endpose/right_endpose"][:])
        left_grip = np.asarray(f["endpose/left_gripper"][:], dtype=np.float32).reshape(-1, 1)
        right_grip = np.asarray(f["endpose/right_gripper"][:], dtype=np.float32).reshape(-1, 1)
        left_arm = np.asarray(f["joint_action/left_arm"][:], dtype=np.float32)
        right_arm = np.asarray(f["joint_action/right_arm"][:], dtype=np.float32)
        t = left_eef.shape[0]
        state = np.concatenate(
            [left_eef, left_grip, right_eef, right_grip, left_arm, right_arm], axis=1
        )
        action = np.concatenate([left_eef, left_grip, right_eef, right_grip], axis=1)
        if state.shape[1] != STATE_DIM or action.shape[1] != ACTION_DIM:
            raise ValueError(f"bad dims state={state.shape} action={action.shape} src={src}")

        cam_frames = {}
        hw = None
        for name, hdf_key in CAMERAS.items():
            frames = []
            rgb = f[f"observation/{hdf_key}/rgb"]
            for i in range(t):
                frames.append(jpeg_to_bgr(rgb[i]))
            cam_frames[name] = frames
            hw = frames[0].shape[:2]

    for name, frames in cam_frames.items():
        write_mp4(
            videos[name],
            frames,
            fps=fps,
            encoder=job.get("encoder", "libx264"),
            gpu_id=job.get("gpu_id"),
        )

    parquet.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "observation.state": [row.astype(np.float32) for row in state],
            "action": [row.astype(np.float32) for row in action],
            "timestamp": (np.arange(t, dtype=np.float32) / fps),
            "frame_index": np.arange(t, dtype=np.int64),
            "episode_index": np.full(t, ep_idx, dtype=np.int64),
            "index": np.arange(t, dtype=np.int64),  # rewritten later if needed
            "task_index": np.zeros(t, dtype=np.int64),
            "annotation.human.task_description": np.zeros(t, dtype=np.int64),
            "next.done": np.zeros(t, dtype=bool),
        }
    )
    df.loc[df.index[-1], "next.done"] = True
    tmp_parquet = parquet.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_parquet, index=False)
    tmp_parquet.replace(parquet)
    return {
        "ep_idx": ep_idx,
        "length": t,
        "instruction": instruction,
        "skipped": False,
        "hw": hw,
        "split": job["split"],
    }


def iter_source_episodes(official_root: Path, task: str) -> list[dict]:
    jobs = []
    mapping = [
        ("clean", official_root / "clean" / task / "aloha-agilex_clean_50"),
        ("randomized", official_root / "randomized" / task / "aloha-agilex_randomized_500"),
    ]
    ep_idx = 0
    fallback = task.replace("_", " ")
    for split, src_root in mapping:
        data_dir = src_root / "data"
        if not data_dir.is_dir():
            continue
        files = sorted(
            data_dir.glob("episode*.hdf5"),
            key=lambda p: int(p.stem.replace("episode", "")),
        )
        for hdf5 in files:
            n = hdf5.stem.replace("episode", "")
            instr = load_instruction(src_root / "instructions" / f"episode{n}.json", fallback)
            jobs.append(
                {
                    "hdf5": str(hdf5),
                    "split": split,
                    "src_ep": int(n),
                    "ep_idx": ep_idx,
                    "instruction": instr,
                    "fps": FPS,
                }
            )
            ep_idx += 1
    return jobs


def rewrite_global_index(out_root: Path, episodes: list[dict]) -> int:
    cursor = 0
    for rec in episodes:
        ep_idx = rec["ep_idx"]
        chunk = ep_idx // 1000
        parquet = out_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet)
        n = len(df)
        df["index"] = np.arange(cursor, cursor + n, dtype=np.int64)
        df["episode_index"] = ep_idx
        df.to_parquet(parquet, index=False)
        rec["length"] = n
        cursor += n
    return cursor


def write_meta(out_root: Path, task: str, episodes: list[dict], hw: tuple[int, int] | None) -> None:
    height, width = hw if hw else (240, 320)
    tasks = []
    task_to_idx = {}
    for rec in episodes:
        text = rec["instruction"]
        if text not in task_to_idx:
            task_to_idx[text] = len(tasks)
            tasks.append(text)
        rec["task_index"] = task_to_idx[text]

    # One parquet rewrite: task_index + global index.
    cursor = 0
    for rec in episodes:
        ep_idx = rec["ep_idx"]
        chunk = ep_idx // 1000
        parquet = out_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet)
        n = len(df)
        df["task_index"] = rec["task_index"]
        df["annotation.human.task_description"] = rec["task_index"]
        df["index"] = np.arange(cursor, cursor + n, dtype=np.int64)
        df["episode_index"] = ep_idx
        df.to_parquet(parquet, index=False)
        rec["length"] = n
        cursor += n
    total_frames = cursor
    meta = out_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    with open(meta / "tasks.jsonl", "w") as f:
        for i, text in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": text}) + "\n")
    with open(meta / "episodes.jsonl", "w") as f:
        for rec in episodes:
            f.write(
                json.dumps(
                    {
                        "episode_index": rec["ep_idx"],
                        "tasks": [rec["instruction"]],
                        "length": rec["length"],
                        "split": rec.get("split"),
                    }
                )
                + "\n"
            )

    repo_modality = Path(__file__).resolve().parents[1] / "modality.json"
    (meta / "modality.json").write_text(repo_modality.read_text())

    video_info = {
        "video.height": height,
        "video.width": width,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": FPS,
        "video.channels": 3,
        "has_audio": False,
    }
    features = {
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": (
                [f"left_eef_{i}" for i in range(9)]
                + ["left_gripper"]
                + [f"right_eef_{i}" for i in range(9)]
                + ["right_gripper"]
            ),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": (
                [f"left_eef_{i}" for i in range(9)]
                + ["left_gripper"]
                + [f"right_eef_{i}" for i in range(9)]
                + ["right_gripper"]
                + [f"left_arm_{i}" for i in range(6)]
                + [f"right_arm_{i}" for i in range(6)]
            ),
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "annotation.human.task_description": {"dtype": "int64", "shape": [1], "names": None},
    }
    for name in CAMERAS:
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": video_info,
        }

    n_chunks = max((ep["ep_idx"] // 1000) for ep in episodes) + 1 if episodes else 1
    info = {
        "codebase_version": "v2.1",
        "robot_type": "agilex_aloha",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "total_chunks": n_chunks,
        "total_videos": len(episodes) * len(CAMERAS),
        "robotwin_task": task,
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")


def convert_task(
    task: str,
    official_root: Path,
    out_root: Path,
    workers: int,
    max_episodes: int = 0,
    encoder: str = "libx264",
    ngpu: int = 0,
) -> int:
    src_jobs = iter_source_episodes(official_root, task)
    if max_episodes and len(src_jobs) > max_episodes:
        src_jobs = src_jobs[:max_episodes]
    if not src_jobs:
        print(f"[skip] {task}: no hdf5", flush=True)
        return 0
    out_task = out_root / task
    out_task.mkdir(parents=True, exist_ok=True)
    jobs = []
    for j in src_jobs:
        j = dict(j)
        j["out_root"] = str(out_task)
        j["encoder"] = encoder
        if encoder == "h264_nvenc" and ngpu > 0:
            j["gpu_id"] = int(j["ep_idx"]) % ngpu
        jobs.append(j)
    results = []
    print(f"[task] {task} episodes={len(jobs)} -> {out_task}", flush=True)
    if workers <= 1:
        for job in jobs:
            results.append(convert_episode(job))
            if len(results) % 25 == 0:
                print(f"  {task} {len(results)}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(convert_episode, job) for job in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    print(f"[fail] {task} {i}/{len(jobs)}: {exc}", flush=True)
                    raise
                if i % 25 == 0 or i == len(futs):
                    print(f"  {task} {i}/{len(jobs)}", flush=True)
    results.sort(key=lambda r: r["ep_idx"])
    hw = None
    for rec, job in zip(results, jobs):
        rec["split"] = job["split"]
        rec["instruction"] = job["instruction"]
        if rec.get("hw"):
            hw = tuple(rec["hw"])
    write_meta(out_task, task, results, hw)
    print(f"[done] {task} n={len(results)} frames={sum(r['length'] for r in results)}", flush=True)
    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("/DRoboticsAILab/vepfs_20260622/licheng.yang/datasets/robotwin2.0_official"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/DRoboticsAILab/vepfs_20260622/licheng.yang/datasets/robotwin2.0_gr00t_lerobot"),
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-episodes-per-task", type=int, default=0, help="0 = all")
    parser.add_argument("--encoder", choices=["libx264", "h264_nvenc"], default="libx264")
    parser.add_argument("--ngpu", type=int, default=0, help="round-robin NVENC GPU count")
    args = parser.parse_args()

    tasks = args.tasks
    if not tasks:
        tasks = sorted(
            d.name
            for d in (args.official_root / "clean").iterdir()
            if d.is_dir() and (d / "aloha-agilex_clean_50").is_dir()
        )
    args.out_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for task in tasks:
        n = convert_task(
            task,
            args.official_root,
            args.out_root,
            args.workers,
            max_episodes=args.max_episodes_per_task,
            encoder=args.encoder,
            ngpu=args.ngpu,
        )
        total += n
        names = sorted(
            p.name
            for p in args.out_root.iterdir()
            if p.is_dir() and (p / "meta" / "info.json").exists()
        )
        (args.out_root / "dataset_paths.txt").write_text("\n".join(names) + "\n")
    names = sorted(
        p.name
        for p in args.out_root.iterdir()
        if p.is_dir() and (p / "meta" / "info.json").exists()
    )
    (args.out_root / "dataset_paths.txt").write_text("\n".join(names) + "\n")
    print(f"ALL_DONE tasks={len(names)} episodes={total}", flush=True)


if __name__ == "__main__":
    # Avoid OpenCV/ffmpeg thread pile-up in workers.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    main()
