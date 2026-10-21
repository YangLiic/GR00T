# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 modality config for D-Robotics AR5 dual-arm real-robot tasks.

Registers under EmbodimentTag.NEW_EMBODIMENT. Load this file via
``--modality-config-path``; do not import it together with the RoboTwin
ALOHA config (both claim NEW_EMBODIMENT).

Action/state: 16-D absolute joints
  left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1).
Cameras: head_camera + left_wrist_camera + right_wrist_camera.
Action chunk: 40 steps (N1.7 pretrained horizon).

Supervision of the action chunk is controlled at launch time by
``--copy-state-to-action`` (OpenPI ``copy_agent_pos_to_actions``). When
that flag is on, the loader reads future proprio from observation.state
instead of the VR ``action`` column. Do not bake original_key into
dataset modality.json; leave the switch in the training config.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


ACTION_HORIZON = 40

JOINT_KEYS = [
    "left_arm",
    "left_gripper",
    "right_arm",
    "right_gripper",
]

ar5_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head_camera", "left_wrist_camera", "right_wrist_camera"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=list(JOINT_KEYS),
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=list(JOINT_KEYS),
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_arm",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_gripper",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_arm",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_gripper",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(ar5_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
