# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: E402  # portable-runtime site setup must precede third-party imports
# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

from pathlib import Path
import site
import sys


# The RoboTTT worktree shares a venv whose editable install points at the baseline.
# Resolve imports from this checkout rather than silently training the old model.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Portable CPython + PYTHONPATH=site-packages does not process .pth files.
# setuptools' distutils shim (needed by DeepSpeed on 3.12) lives in one.
_venv_site = Path(__file__).resolve().parents[2] / ".venv" / "lib" / "python3.12" / "site-packages"
if _venv_site.is_dir():
    site.addsitedir(str(_venv_site))

import json
import os

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.use_percentiles = ft_config.use_percentiles
    if (ft_config.shortest_image_edge is None) != (ft_config.crop_fraction is None):
        raise ValueError("shortest_image_edge and crop_fraction must be set together")
    if ft_config.shortest_image_edge is not None:
        config.model.shortest_image_edge = ft_config.shortest_image_edge
        config.model.crop_fraction = ft_config.crop_fraction
        config.model.image_crop_size = None
        config.model.image_target_size = None
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    # 训练时 flow matching 时间采样的 Beta 形状参数。默认留空 = 用 config 默认值
    # (1.5, 1.0)，不要给这两个 env 设默认；改时间分布的消融（dense-jump 那类）只在
    # 那一次训练的启动脚本里显式 export，这样默认配置永远是原分布。
    # 与 ROBOTWIN_BGR2RGB 一样走 env 而不是 CLI flag，值会落进 checkpoint config.json。
    _beta_alpha = os.environ.get("GR00T_NOISE_BETA_ALPHA")
    _beta_beta = os.environ.get("GR00T_NOISE_BETA_BETA")
    if _beta_alpha:
        config.model.noise_beta_alpha = float(_beta_alpha)
    if _beta_beta:
        config.model.noise_beta_beta = float(_beta_beta)
    # Path must contain "nvidia/Cosmos-Reason2" (get_backbone_cls). Prefer a
    # local tree so A800/Licheng can load offline without a Hub snapshot.
    cosmos_env = os.environ.get("GR00T_COSMOS_MODEL_NAME")
    cosmos_local = (
        Path(__file__).resolve().parents[2] / "checkpoints" / "nvidia" / "Cosmos-Reason2-2B"
    )
    if cosmos_env:
        config.model.model_name = cosmos_env
    elif (cosmos_local / "config.json").exists():
        config.model.model_name = str(cosmos_local)
    else:
        config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        config.training.transformers_local_files_only = True
    config.model.backbone_trainable_params_fp32 = True
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.types import ActionRepresentation

    action_mod = MODALITY_CONFIGS[embodiment_tag]["action"]
    action_cfgs = action_mod.action_configs or []
    config.model.use_relative_action = any(
        cfg.rep == ActionRepresentation.RELATIVE for cfg in action_cfgs
    )
    data_action_horizon = len(action_mod.delta_indices)
    if data_action_horizon > config.model.action_horizon:
        print(
            f"Raising model.action_horizon {config.model.action_horizon} -> {data_action_horizon} "
            "to match modality delta_indices",
            flush=True,
        )
        config.model.action_horizon = data_action_horizon
    print(
        f"use_relative_action={config.model.use_relative_action} "
        f"action_horizon={data_action_horizon} "
        f"model.action_horizon={config.model.action_horizon} "
        f"copy_state_to_action={ft_config.copy_state_to_action}",
        f"bgr_to_rgb={os.environ.get('ROBOTWIN_BGR2RGB', '0')}",
        f"noise_beta=({config.model.noise_beta_alpha}, {config.model.noise_beta_beta})",
        flush=True,
    )

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.ds_weights_alpha = ft_config.ds_weights_alpha
    config.data.copy_state_to_action = ft_config.copy_state_to_action
    # BGR->RGB 统一用 ROBOTWIN_BGR2RGB env 控制（与 openpi 训练/推理、GR00T 推理一致），
    # 不再用单独的 --bgr-to-rgb CLI flag。RoboTwin LeRobot 数据是 BGR 时设 1。
    config.data.bgr_to_rgb = os.environ.get("ROBOTWIN_BGR2RGB", "0").lower() in ("1", "true", "yes")

    config.training.save_only_model = ft_config.save_only_model
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    if ft_config.robottt_enabled:
        config.model.action_horizon = data_action_horizon
        for key in ft_config.__dataclass_fields__:
            if key.startswith("robottt_"):
                setattr(config.model, key, getattr(ft_config, key))
        if config.model.robottt_train_mode not in {"memory", "posttrain"}:
            raise ValueError("RoboTTT mode must be memory or posttrain")
        if config.model.robottt_train_mode == "memory":
            config.model.tune_llm = False
            config.model.tune_visual = False
            config.model.tune_projector = False
            config.model.tune_diffusion_model = False
            config.model.tune_vlln = False
        config.training.use_ddp = True
        config.training.gradient_checkpointing = False
        config.training.remove_unused_columns = False
        config.training.lr_scheduler_type = "cosine"
        if config.model.robottt_tbptt_steps < 1 or config.model.robottt_context_length < 1:
            raise ValueError("RoboTTT context length and TBPTT steps must be positive")
        if config.training.fp16:
            raise ValueError(
                "RoboTTT reference training uses fp32 fast weights with fp32/bf16 outer computation"
            )

    run(config)
