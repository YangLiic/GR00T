"""Chronological, episode-bounded windows over the established LeRobot processor."""

import copy
import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def episode_windows(length, sequence_steps, stride):
    """Indices never cross an episode or reorder time; incomplete tails are kept."""
    if sequence_steps < 1 or stride < 1:
        raise ValueError("sequence_steps and stride must be positive")
    for start in range(0, length, sequence_steps * stride):
        yield list(range(start, min(start + sequence_steps * stride, length), stride))


class RoboTTTSequenceDataset(Dataset):
    def __init__(self, mixture, config):
        self.mixture = mixture
        self.datasets = mixture.datasets
        self.seed = mixture.seed
        self.stride = config.robottt_chunk_stride or config.action_horizon
        self.sequence_steps = math.ceil(config.robottt_context_length / self.stride)
        if self.sequence_steps < 1 or self.stride < 1:
            raise ValueError("RoboTTT context/stride must be positive")
        self.loss_mask_column = config.robottt_loss_mask_column
        self.windows = []
        # Reuse per-episode loading, normalization, action representation and RGB
        # handling. Do not consume the mixture's shuffled individual timesteps.
        for ds_idx, ds in enumerate(self.datasets):
            for ep in range(len(ds.episode_loader)):
                length = ds.get_effective_episode_length(ep)
                for indices in episode_windows(length, self.sequence_steps, self.stride):
                    self.windows.append((ds_idx, ep, indices))
        if not self.windows:
            raise ValueError("No episode is long enough to contain one complete action chunk")
        weights = np.asarray(mixture.weights, dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Dataset mixture weights must be finite, nonnegative and nonzero")
        counts = np.bincount([ds for ds, _, _ in self.windows], minlength=len(weights))
        if ((counts == 0) & (weights > 0)).any():
            raise ValueError("A positive-weight dataset has no valid sequence windows")
        self.sampling_weights = torch.tensor(
            [weights[ds] / counts[ds] for ds, _, _ in self.windows], dtype=torch.double
        )
        self._cached_key = None
        self._cached_episode = None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        ds_idx, ep, indices = self.windows[index]
        ds = self.datasets[ds_idx]
        if self._cached_key != (ds_idx, ep):
            self._cached_episode = ds.episode_loader[ep]
            self._cached_key = (ds_idx, ep)
        episode = self._cached_episode
        frames = []
        for step in indices:
            frame = ds.get_datapoint(episode, step)
            frame["robottt_valid"] = np.array(True)
            weight = 1.0
            if self.loss_mask_column:
                if self.loss_mask_column not in episode.columns:
                    raise ValueError(f"Missing loss-mask column {self.loss_mask_column!r}")
                weight = float(episode.iloc[step][self.loss_mask_column])
                if not np.isfinite(weight) or not 0 <= weight <= 1:
                    raise ValueError("Action loss-mask weights must be finite and in [0, 1]")
            # A loss-masked correction/video context still writes to memory.
            frame["action_mask"] = frame["action_mask"] * weight
            frames.append(frame)
        while len(frames) < self.sequence_steps:
            frame = copy.deepcopy(frames[-1])
            frame["robottt_valid"] = np.array(False)
            frame["action_mask"] = np.zeros_like(frame["action_mask"])
            frames.append(frame)
        for step, frame in enumerate(frames):
            frame["robottt_reset"] = np.array(step == 0)
        return {"robottt_sequence": frames}

    def reset_seed(self, seed):
        self.seed = seed

    def get_dataset_statistics(self):
        return self.mixture.get_dataset_statistics()

    def get_initial_actions(self):
        return self.mixture.get_initial_actions()


class RoboTTTSequenceCollator:
    def __init__(self, frame_collator):
        self.frame_collator = frame_collator

    def __call__(self, samples):
        lengths = {len(item["robottt_sequence"]) for item in samples}
        if len(lengths) != 1:
            raise ValueError("All sequences in a batch must use the same padded length")
        frames = [
            self.frame_collator([item["robottt_sequence"][t] for item in samples])
            for t in range(lengths.pop())
        ]
        return {"robottt_sequence": frames}


class RoboTTTSequenceSampler(Sampler):
    """Rank-symmetric weighted sampling; Accelerate shards the resulting batches.

    Epoch seed is explicit for deterministic resume. Replacement is intentional:
    dataset-level mixture weights must not become proportional to episode length.
    """

    def __init__(self, dataset, seed):
        self.weights = dataset.sampling_weights
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(
            torch.multinomial(
                self.weights, len(self), replacement=True, generator=generator
            ).tolist()
        )
