"""CPU-only DDP smoke: torchrun --standalone --nproc_per_node=2 this_file.py."""

import os
from pathlib import Path
import tempfile

from gr00t.data.dataset.robottt_sequence_dataset import RoboTTTSequenceCollator
from gr00t.experiment.robottt_trainer import RoboTTTTrainer
import numpy as np
from tests.robottt.test_integration import TinyVLA, frame
import torch
import torch.distributed as dist
from transformers import TrainingArguments


def collate(batch):
    return {
        "inputs": {key: torch.from_numpy(np.stack([f[key] for f in batch])) for key in batch[0]}
    }


def main():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = TinyVLA()
    frames = [frame(reset=t == 0) for t in range(3)]
    sample = {"robottt_sequence": [{k: v[0].numpy() for k, v in f.items()} for f in frames]}
    args = TrainingArguments(
        output_dir=str(Path(tempfile.gettempdir()) / "robottt-ddp-smoke"),
        use_cpu=True,
        ddp_backend="gloo",
        max_steps=2,
        per_device_train_batch_size=1,
        report_to=[],
        save_strategy="no",
        remove_unused_columns=False,
        learning_rate=1e-3,
        disable_tqdm=True,
        ddp_find_unused_parameters=True,
    )
    trainer = RoboTTTTrainer(
        model=model,
        args=args,
        train_dataset=[sample] * 4,
        data_collator=RoboTTTSequenceCollator(collate),
    )
    before = model.action_head.model.transformer_blocks[0].robottt.w1_init.detach().clone()
    trainer.train()
    after = model.action_head.model.transformer_blocks[0].robottt.w1_init.detach()
    gathered = [torch.empty_like(after) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, after)
    assert all(torch.equal(after, value) for value in gathered)
    assert not torch.equal(before, after)
    print(
        f"ROBOTTT_DDP_OK rank={os.environ['RANK']} optimizer_steps={trainer.state.global_step}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
