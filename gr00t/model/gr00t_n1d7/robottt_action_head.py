"""Opt-in RoboTTT action head; original encoders, decoder and Euler/RTC path retained."""

from contextlib import contextmanager

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.modules.ttt_memory import TTTContext, TTTMemory


class RoboTTTActionHead(Gr00tN1d7ActionHead):
    def __init__(self, config):
        super().__init__(config)
        self.register_tokens = nn.Parameter(
            torch.empty(config.robottt_num_register_tokens, self.input_embedding_dim)
        )
        nn.init.normal_(self.register_tokens, std=0.02)
        for idx, block in enumerate(self.model.transformer_blocks):
            block.robottt = TTTMemory(
                block.dim,
                config.robottt_hidden_dim,
                idx,
                config.robottt_inner_lr,
                config.robottt_gate_init,
                config.robottt_rope_theta,
            )
        self._train_context = None
        self._runtime_context = None
        self._active_context = None
        self._candidate_context = None
        self._transactional_inference = False
        self.model.register_forward_pre_hook(self._inject_memory, with_kwargs=True)

    def _inject_memory(self, module, args, kwargs):
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            raise NotImplementedError(
                "RoboTTT requires explicit fast-state inputs/outputs before ONNX/TensorRT export"
            )
        if self._active_context is None:
            raise RuntimeError("RoboTTT DiT requires an explicit sequence/rollout context")
        tokens = kwargs["hidden_states"]
        registers = (
            self.register_tokens.to(tokens.dtype).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        )
        kwargs["hidden_states"] = torch.cat((registers, tokens), dim=1)
        context = (
            self._active_context.fork(detach=True)
            if self._transactional_inference
            else self._active_context
        )
        self._candidate_context = context
        kwargs["robottt_context"] = context
        return args, kwargs

    @contextmanager
    def _use_context(self, context, transactional=False):
        if self._active_context is not None:
            raise RuntimeError(
                "Concurrent/reentrant calls cannot share a RoboTTT rollout; use one policy per stream"
            )
        self._active_context = context
        self._transactional_inference = transactional
        try:
            yield
        finally:
            self._active_context = None
            self._transactional_inference = False

    def reset_memory(self):
        """Call at every new episode/task; never carry memory between unrelated clients."""
        self._runtime_context = None
        self._train_context = None
        self._candidate_context = None

    def detach_training_memory(self):
        if self._train_context is not None:
            self._train_context.detach()

    def forward(self, backbone_output, action_input):
        if "robottt_valid" not in action_input:
            raise ValueError(
                "RoboTTT training requires episode sequences; use the RoboTTT dataset/trainer"
            )
        valid = action_input["robottt_valid"].reshape(-1).bool()
        reset = action_input["robottt_reset"].reshape(-1).bool()
        if reset.any():
            if not reset.all():
                raise ValueError(
                    "Training sequences must start together; partial resets are not supported"
                )
            self._train_context = TTTContext(
                len(self.model.transformer_blocks), len(valid), valid.device
            )
        context = self._train_context
        if context is None or len(context.offsets) != len(valid):
            raise ValueError("Missing sequence reset, or batch size changed inside a sequence")
        context.write_mask = valid
        # Parent forward reshapes state in-place: protect the caller's batch dictionary.
        prepared = BatchFeature(data=dict(action_input))
        with self._use_context(context):
            output = BatchFeature(data=super().forward(backbone_output, prepared))
        count = action_input.action_mask.sum()
        output["loss_sum"] = output.action_loss.sum()
        output["loss_count"] = count
        context.advance(self.register_tokens.shape[0] + 1 + action_input.action.shape[1])
        return output

    def get_action_with_features(
        self,
        backbone_features,
        state_features,
        embodiment_id,
        backbone_output,
        action_input,
        options=None,
    ):
        batch_size = backbone_features.shape[0]
        if self._runtime_context is None:
            self._runtime_context = TTTContext(
                len(self.model.transformer_blocks), batch_size, backbone_features.device
            )
        context = self._runtime_context
        if len(context.offsets) != batch_size or context.offsets.device != backbone_features.device:
            raise ValueError(
                "Environment batch/device changed; call policy.reset() before reusing memory"
            )
        # All denoising evaluations start from the same preceding observation memory.
        # Commit only the final candidate: flow steps do not multiply physical history.
        self._candidate_context = None
        with self._use_context(context, transactional=True):
            result = super().get_action_with_features(
                backbone_features,
                state_features,
                embodiment_id,
                backbone_output,
                action_input,
                options,
            )
        if self._candidate_context is None:
            raise RuntimeError("No denoising evaluation was performed")
        self._runtime_context = self._candidate_context
        self._runtime_context.advance(
            self.register_tokens.shape[0] + state_features.shape[1] + self.action_horizon
        )
        self._runtime_context.detach()
        return result


def is_memory_parameter(name):
    return ".robottt." in name or name.endswith("register_tokens")


def initialize_missing_memory(model, missing_keys):
    """Explicitly initialize new modules when upgrading a baseline checkpoint.

    Partial memory checkpoints are rejected: silently reinitializing a previously
    trained memory would produce apparently valid but broken deployment behavior.
    """
    missing = set(missing_keys)
    allowed = {name for name, _ in model.named_parameters() if is_memory_parameter(name)}
    unexpected_missing = missing - allowed
    if unexpected_missing:
        raise RuntimeError(
            f"Non-memory checkpoint parameters are missing: {sorted(unexpected_missing)}"
        )
    if missing and missing != allowed:
        raise RuntimeError(
            "Partially missing RoboTTT memory checkpoint; refusing to reinitialize trained parameters"
        )
    if missing:
        for module in model.modules():
            if isinstance(module, TTTMemory):
                module.reset_parameters()
        nn.init.normal_(model.action_head.register_tokens, std=0.02)


def configure_robottt_training(model):
    mode = model.config.robottt_train_mode
    if mode not in {"memory", "posttrain"}:
        raise ValueError("robottt_train_mode must be 'memory' or 'posttrain'")
    if mode == "memory":
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if is_memory_parameter(name):
                param.requires_grad_(True)
    # posttrain respects existing tune_llm/tune_visual/projector options. The
    # launcher sets all of them True for the paper's all-parameter post-training.
