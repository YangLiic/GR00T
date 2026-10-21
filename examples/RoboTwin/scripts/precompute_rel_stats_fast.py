#!/usr/bin/env python3
"""Episode-parallel GR00T relative stats. Official generate_rel_stats is ~1.5s/ep
and would take ~20h for 50 tasks; this keeps the same math and fingerprint.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

_WORKER = {}


def _init_worker(dataset_path: str, embodiment_value: str, action_key: str, modality_py: str) -> None:
    sys.path.append(str(Path(modality_py).parent))
    import importlib

    importlib.import_module(Path(modality_py).stem)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import RelativeActionLoader

    tag = EmbodimentTag(embodiment_value)
    _WORKER["loader"] = RelativeActionLoader(dataset_path, tag, action_key)


def _one_episode(ep_id: int):
    return _WORKER["loader"].load_relative_actions(ep_id)


def compute_rel_key(dataset_path: Path, embodiment_tag, action_key: str, modality_py: Path, workers: int) -> dict:
    from gr00t.data.stats import RelativeActionLoader

    n = len(RelativeActionLoader(dataset_path, embodiment_tag, action_key))
    ctx_workers = max(1, workers)
    trajectories: list[np.ndarray] = []
    with ProcessPoolExecutor(
        max_workers=ctx_workers,
        initializer=_init_worker,
        initargs=(str(dataset_path), embodiment_tag.value, action_key, str(modality_py)),
    ) as ex:
        for chunk in ex.map(_one_episode, range(n), chunksize=4):
            trajectories.extend(chunk)
    stacked = np.stack(trajectories, axis=0)
    return {
        "max": np.max(stacked, axis=0),
        "min": np.min(stacked, axis=0),
        "q01": np.quantile(stacked, 0.01, axis=0),
        "mean": np.mean(stacked, axis=0),
        "std": np.std(stacked, axis=0),
        "q99": np.quantile(stacked, 0.99, axis=0),
    }


def _ensure_modality(modality_py: Path) -> None:
    sys.path.append(str(modality_py.parent))
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag

    if EmbodimentTag.NEW_EMBODIMENT.value not in MODALITY_CONFIGS:
        import importlib

        importlib.import_module(modality_py.stem)


def process_task(dataset_path: Path, modality_py: Path, workers: int) -> None:
    _ensure_modality(modality_py)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import (
        MODALITY_CONFIGS,
        STATS_FINGERPRINTS_KEY,
        _compute_relative_action_fingerprint,
        _dump_stats_cache_atomic,
        _load_stats_cache,
        generate_stats,
        to_json_serializable,
        LE_ROBOT_REL_STATS_FILENAME,
    )
    from gr00t.data.types import ActionRepresentation

    tag = EmbodimentTag.NEW_EMBODIMENT
    generate_stats(dataset_path)
    action_config = MODALITY_CONFIGS[tag.value]["action"]
    action_keys = [
        key
        for key, cfg in zip(action_config.modality_keys, action_config.action_configs)
        if cfg.rep == ActionRepresentation.RELATIVE
    ]
    stats_path = dataset_path / LE_ROBOT_REL_STATS_FILENAME
    stats = _load_stats_cache(stats_path)
    fingerprints = stats.setdefault(STATS_FINGERPRINTS_KEY, {})
    for action_key in sorted(action_keys):
        expected_fp = _compute_relative_action_fingerprint(tag, action_key)
        if action_key in stats and fingerprints.get(action_key) == expected_fp:
            print(f"skip rel {dataset_path.name} {action_key}", flush=True)
            continue
        print(f"rel {dataset_path.name} {action_key} workers={workers}", flush=True)
        stats[action_key] = compute_rel_key(dataset_path, tag, action_key, modality_py, workers)
        fingerprints[action_key] = expected_fp
        _dump_stats_cache_atomic(stats_path, to_json_serializable(dict(stats)))
    print(f"done {dataset_path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--modality-config-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--once", action="store_true", help="process current tasks and exit")
    args = parser.parse_args()

    import time

    while True:
        did = 0
        for info in sorted(args.out_root.glob("*/meta/info.json")):
            task_dir = info.parents[1]
            rel = task_dir / "meta" / "relative_stats.json"
            st = task_dir / "meta" / "stats.json"
            if st.exists() and rel.exists():
                continue
            print(f"[task] {task_dir.name}", flush=True)
            process_task(task_dir, args.modality_config_path, args.workers)
            did += 1
        convert_pid_file = args.out_root / "convert.pid"
        convert_alive = False
        if convert_pid_file.exists():
            try:
                os.kill(int(convert_pid_file.read_text().strip()), 0)
                convert_alive = True
            except OSError:
                convert_alive = False
        n_info = len(list(args.out_root.glob("*/meta/info.json")))
        n_rel = len(list(args.out_root.glob("*/meta/relative_stats.json")))
        print(f"info={n_info} rel_stats={n_rel} convert_alive={convert_alive}", flush=True)
        if args.once or (not convert_alive and n_rel >= n_info and n_info >= 50):
            print("ALL_STATS_DONE", flush=True)
            return
        if not convert_alive and n_info < 50 and did == 0:
            raise SystemExit(f"convert gone, incomplete info={n_info} rel={n_rel}")
        time.sleep(20)


if __name__ == "__main__":
    main()
