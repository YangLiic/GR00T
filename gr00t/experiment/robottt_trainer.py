"""Actual truncated BPTT: backward and release each segment before advancing.

Merely detaching a recurrent tensor while accumulating all losses until the end
does NOT bound activation memory. This trainer performs one backward per segment
and one optimizer step per sequence batch, using the existing HF outer loop.
"""

import torch
import torch.distributed as dist
from transformers import Trainer

from gr00t.experiment.trainer import Gr00tTrainer


class RoboTTTTrainer(Gr00tTrainer):
    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if hasattr(dataset, "sampling_weights"):
            from gr00t.data.dataset.robottt_sequence_dataset import RoboTTTSequenceSampler

            return RoboTTTSequenceSampler(dataset, self.args.seed)
        return super()._get_train_sampler(train_dataset)

    def get_train_dataloader(self):
        # Finite episode windows use the standard distributed sampler, not the
        # baseline IterableDataset/shard schedule.
        return Trainer.get_train_dataloader(self)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.is_deepspeed_enabled or self.is_fsdp_enabled or self.use_apex:
            raise NotImplementedError(
                "Reference RoboTTT TBPTT supports single device / DDP with bf16, not DeepSpeed/FSDP/Apex"
            )
        if self.args.n_gpu > 1:
            raise ValueError("Use torchrun DDP, not torch.nn.DataParallel, for recurrent memory")
        model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        bare = self.accelerator.unwrap_model(model)
        head = bare.action_head
        if not bare.config.robottt_enabled:
            raise ValueError("RoboTTTTrainer requires a RoboTTT checkpoint/config")
        steps = bare.config.robottt_tbptt_steps
        if steps < 1:
            raise ValueError("robottt_tbptt_steps must be positive")
        frames = inputs["robottt_sequence"]  # keep the full sequence on CPU
        denominator = sum(frame["inputs"]["action_mask"].float().sum() for frame in frames)
        denominator = denominator.to(self.args.device)
        if dist.is_initialized():
            dist.all_reduce(denominator)
            denominator = denominator / dist.get_world_size()  # DDP averages gradients
        if denominator.item() <= 0:
            raise ValueError(
                "A training sequence batch needs at least one supervised action target"
            )
        total = torch.zeros((), device=self.args.device)
        accumulation = getattr(
            self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps
        )
        try:
            for start in range(0, len(frames), steps):
                segment = [
                    self._prepare_inputs(frame)["inputs"] for frame in frames[start : start + steps]
                ]
                with self.compute_loss_context_manager():
                    output = model(
                        inputs={"robottt_sequence": segment, "robottt_normalizer": denominator}
                    )
                    loss = output["loss"] / accumulation
                self.accelerator.backward(loss)
                total += loss.detach()
                # The values carry across boundaries; only their graph is cut.
                head.detach_training_memory()
                del output, loss, segment
        finally:
            # A new sequence starts from learned W0, not the previous dataset item.
            head._train_context = None
        return total
