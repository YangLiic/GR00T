import copy
import importlib
from types import SimpleNamespace

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.robottt_sequence_dataset import (
    RoboTTTSequenceCollator,
    RoboTTTSequenceDataset,
    RoboTTTSequenceSampler,
    episode_windows,
)
from gr00t.experiment.robottt_trainer import RoboTTTTrainer
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.robottt_action_head import (
    RoboTTTActionHead,
    configure_robottt_training,
    initialize_missing_memory,
)
import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoModel, TrainingArguments
from transformers.feature_extraction_utils import BatchFeature


def tiny_config(**overrides):
    args = dict(
        robottt_enabled=True,
        robottt_num_register_tokens=2,
        robottt_hidden_dim=6,
        robottt_tbptt_steps=2,
        input_embedding_dim=8,
        hidden_size=8,
        backbone_embedding_dim=8,
        max_state_dim=4,
        max_action_dim=3,
        max_num_embodiments=2,
        action_horizon=2,
        num_inference_timesteps=3,
        state_dropout_prob=0.0,
        use_alternate_vl_dit=False,
        use_vlln=False,
        vl_self_attention_cfg=None,
        diffusion_model_cfg=dict(
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=4,
            output_dim=8,
            dropout=0.0,
            final_dropout=False,
            positional_embeddings=None,
            norm_type="ada_norm",
            interleave_self_attention=True,
        ),
    )
    args.update(overrides)
    return Gr00tN1d7Config(**args)


def frame(reset=False, valid=True):
    return dict(
        state=torch.randn(1, 1, 4),
        action=torch.randn(1, 2, 3),
        action_mask=torch.ones(1, 2, 3) * valid,
        embodiment_id=torch.zeros(1, dtype=torch.long),
        robottt_valid=torch.tensor([valid]),
        robottt_reset=torch.tensor([reset]),
        features=torch.randn(1, 3, 8),
    )


def backbone(data):
    return BatchFeature(
        data={
            "backbone_features": data["features"],
            "backbone_attention_mask": torch.ones(1, 3, dtype=torch.bool),
        }
    )


def test_baseline_upgrade_preserves_all_existing_weights():
    base = Gr00tN1d7ActionHead(tiny_config(robottt_enabled=False))
    upgraded = RoboTTTActionHead(tiny_config())
    missing = upgraded.load_state_dict(base.state_dict(), strict=False)
    assert not missing.unexpected_keys
    fake_model = nn.Module()
    fake_model.action_head = upgraded
    initialize_missing_memory(fake_model, ["action_head." + key for key in missing.missing_keys])
    for key, tensor in base.state_dict().items():
        torch.testing.assert_close(upgraded.state_dict()[key], tensor, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="Partially"):
        initialize_missing_memory(fake_model, ["action_head.register_tokens"])


def test_outer_action_loss_reaches_memory_when_original_policy_frozen():
    model = nn.Module()
    model.action_head = RoboTTTActionHead(tiny_config(robottt_train_mode="memory"))
    model.config = model.action_head.config
    configure_robottt_training(model)
    model.train()
    loss = 0
    for t in range(2):
        data = frame(reset=t == 0)
        loss = loss + model.action_head(backbone(data), BatchFeature(data=data)).loss
    loss.backward()
    mem = model.action_head.model.transformer_blocks[0].robottt
    for parameter in (mem.w1_init, mem.k_proj.weight, mem.v_proj.weight, mem.gate):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert model.action_head.action_decoder.layer1.W.grad is None


@pytest.mark.parametrize("alternate", [False, True])
def test_inference_resets_and_commits_once_per_observation(alternate):
    head = RoboTTTActionHead(tiny_config(use_alternate_vl_dit=alternate)).eval()
    data = frame()
    bb = backbone(data)
    data.pop("action")  # an action input at inference activates the existing RTC path
    bb["image_mask"] = torch.tensor([[True, False, True]])
    bb["backbone_attention_mask"] = torch.ones(1, 3, dtype=torch.bool)
    with torch.inference_mode():
        torch.manual_seed(5)
        first = head.get_action(
            copy.deepcopy(bb), BatchFeature(data=copy.deepcopy(data))
        ).action_pred
        assert head._runtime_context.offsets.item() == 5  # R=2, state=1, H=2; NOT 3*5
        head.get_action(copy.deepcopy(bb), BatchFeature(data=copy.deepcopy(data)))
        assert head._runtime_context.offsets.item() == 10
        head.reset_memory()
        torch.manual_seed(5)
        repeated = head.get_action(
            copy.deepcopy(bb), BatchFeature(data=copy.deepcopy(data))
        ).action_pred
    torch.testing.assert_close(first, repeated, atol=0, rtol=0)


def test_sequence_windows_and_padding_do_not_cross_episode():
    assert list(episode_windows(7, 3, 2)) == [[0, 2, 4], [6]]

    class FakeDataset:
        def __init__(self):
            self.episode_loader = [list(range(7)), list(range(3))]

        def get_effective_episode_length(self, ep):
            return len(self.episode_loader[ep])

        def get_datapoint(self, episode, step):
            return {"index": np.array(episode[step]), "action_mask": np.ones((2, 3))}

    mixture = SimpleNamespace(datasets=[FakeDataset()], seed=0, weights=[1.0])
    config = SimpleNamespace(
        robottt_chunk_stride=2,
        action_horizon=2,
        robottt_context_length=6,
        robottt_loss_mask_column=None,
    )
    ds = RoboTTTSequenceDataset(mixture, config)
    tail = ds[1]["robottt_sequence"]
    assert [int(f["index"]) for f in tail] == [6, 6, 6]
    assert [bool(f["robottt_valid"]) for f in tail] == [True, False, False]
    assert tail[1]["action_mask"].sum() == 0
    assert all(bool(ds[i]["robottt_sequence"][0]["robottt_reset"]) for i in range(len(ds)))


class TinyVLA(nn.Module):
    forward = Gr00tN1d7.forward

    def __init__(self):
        super().__init__()
        self.config = tiny_config()
        self.action_head = RoboTTTActionHead(self.config)
        self.backbone = nn.Identity()

    def prepare_input(self, data):
        return backbone(data), BatchFeature(data=data)


def test_two_optimizer_steps_with_real_tbptt(tmp_path):
    model = TinyVLA()
    frames = [frame(reset=t == 0) for t in range(3)]
    # The test dataset stores unbatched frames; the ordinary frame collator stacks them.
    sample = {"robottt_sequence": [{k: v[0].numpy() for k, v in f.items()} for f in frames]}

    def collate(batch):
        return {"inputs": {k: torch.from_numpy(np.stack([f[k] for f in batch])) for k in batch[0]}}

    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        max_steps=2,
        per_device_train_batch_size=1,
        report_to=[],
        save_strategy="no",
        remove_unused_columns=False,
        learning_rate=1e-3,
        disable_tqdm=True,
    )
    trainer = RoboTTTTrainer(
        model=model,
        args=args,
        train_dataset=[sample, sample],
        data_collator=RoboTTTSequenceCollator(collate),
    )
    before = model.action_head.model.transformer_blocks[0].robottt.q_proj.weight.detach().clone()
    calls = []
    original = trainer.accelerator.backward

    def backward(loss, **kwargs):
        calls.append(float(loss.detach()))
        return original(loss, **kwargs)

    trainer.accelerator.backward = backward
    trainer.train()
    assert len(calls) == 4  # two segments per sequence, two optimizer updates
    assert model.action_head._train_context is None
    assert not torch.equal(
        before, model.action_head.model.transformer_blocks[0].robottt.q_proj.weight
    )


def test_config_roundtrip_and_state_dict_excludes_rollout(tmp_path):
    config = tiny_config()
    config.save_pretrained(tmp_path)
    loaded = Gr00tN1d7Config.from_pretrained(tmp_path)
    assert loaded.robottt_enabled and loaded.robottt_hidden_dim == 6
    head = RoboTTTActionHead(config)
    data = frame(reset=True)
    output = head(backbone(data), BatchFeature(data=data))
    output.loss.backward()
    clone = RoboTTTActionHead(loaded)
    clone.load_state_dict(head.state_dict(), strict=True)
    assert clone._runtime_context is None and clone._train_context is None


def test_real_auto_model_upgrade_and_checkpoint_reload(
    tmp_path, monkeypatch, load_hf_model_weights
):
    module = importlib.import_module("gr00t.model.gr00t_n1d7.gr00t_n1d7")
    processing = importlib.import_module("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")

    class FakeBackbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.proj = nn.Linear(8, 8)

    monkeypatch.setattr(module, "get_backbone_cls", lambda config: FakeBackbone)
    monkeypatch.setattr(processing, "resolve_vlm_pretrained_path", lambda name: name)
    monkeypatch.setattr(processing, "Gr00tN1d7DataCollator", lambda **kwargs: None)
    baseline = Gr00tN1d7(tiny_config(robottt_enabled=False))
    source = tmp_path / "baseline"
    baseline.save_pretrained(source)
    with load_hf_model_weights():
        upgraded, info = AutoModel.from_pretrained(
            source, robottt_enabled=True, local_files_only=True, output_loading_info=True
        )
        initialize_missing_memory(upgraded, info["missing_keys"])
        assert not info["unexpected_keys"] and not info["mismatched_keys"]
        for key, tensor in baseline.state_dict().items():
            torch.testing.assert_close(upgraded.state_dict()[key], tensor, atol=0, rtol=0)
        destination = tmp_path / "robottt"
        upgraded.save_pretrained(destination)
        restored, info = AutoModel.from_pretrained(
            destination, local_files_only=True, output_loading_info=True
        )
    assert isinstance(restored.action_head, RoboTTTActionHead)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    for key, tensor in upgraded.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor, atol=0, rtol=0)


def test_weighted_sequence_sampler_is_epoch_reproducible():
    dataset = SimpleNamespace(sampling_weights=torch.tensor([0.1, 0.2, 0.7], dtype=torch.double))
    sampler = RoboTTTSequenceSampler(dataset, seed=7)
    assert list(sampler) == list(RoboTTTSequenceSampler(dataset, seed=7))
    sampler.set_epoch(1)
    expected = torch.multinomial(
        dataset.sampling_weights, 3, replacement=True, generator=torch.Generator().manual_seed(8)
    ).tolist()
    assert list(sampler) == expected
