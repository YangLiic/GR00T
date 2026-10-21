"""Reference RoboTTT key/value-binding memory (paper Eqs. 1--3).

Fast tensors are recurrent state, never registered parameters or optimizer state.
The analytic SGD step is differentiable, including second-order meta-gradients,
and also works under torch.inference_mode(), without enabling autograd at serving.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FastWeights:
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor

    def detach(self):
        return FastWeights(*(x.detach() for x in self.tensors()))

    def tensors(self):
        return self.w1, self.b1, self.w2, self.b2


def fast_mlp(x, weights):
    return F.gelu(x @ weights.w1 + weights.b1) @ weights.w2 + weights.b2


def sgd_step(keys, values, weights, lr, write_mask=None):
    """Exactly one SGD step on mean_{tokens,features} ||f(K)-V||^2.

    Batch members have independent memories. There is deliberately no averaging
    over batch, which would make adaptation depend on the number of environments.
    No detach is used here: the outer action loss differentiates through this step.
    """
    pre = keys @ weights.w1 + weights.b1
    hidden = F.gelu(pre)
    error = (hidden @ weights.w2 + weights.b2 - values) * (2 / values.shape[1] / values.shape[2])
    grad_w2 = hidden.transpose(-1, -2) @ error
    grad_b2 = error.sum(dim=1, keepdim=True)
    derivative = 0.5 * (1 + torch.erf(pre / math.sqrt(2)))
    derivative = derivative + pre * torch.exp(-0.5 * pre.square()) / math.sqrt(2 * math.pi)
    back = (error @ weights.w2.transpose(-1, -2)) * derivative
    grads = (keys.transpose(-1, -2) @ back, back.sum(dim=1, keepdim=True), grad_w2, grad_b2)
    if write_mask is not None:
        lr = lr * write_mask.to(keys.dtype).reshape(-1, 1, 1)
    return FastWeights(*(w - lr * g for w, g in zip(weights.tensors(), grads)))


def apply_rope(x, offsets, theta=10000.0):
    """Rotary positions over the flattened temporal token stream, not flow time."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("TTT projection dimension must be even for RoPE")
    inv_freq = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    pos = offsets[:, None] + torch.arange(x.shape[1], device=x.device)[None, :]
    angles = pos.float()[..., None] * inv_freq
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class TTTMemory(nn.Module):
    def __init__(
        self, dim, hidden_dim, layer_index, inner_lr=0.1, gate_init=0.001, rope_theta=10000.0
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layer_index = layer_index
        self.inner_lr = float(inner_lr)
        self.rope_theta = float(rope_theta)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.w1_init = nn.Parameter(torch.empty(dim, hidden_dim))
        self.b1_init = nn.Parameter(torch.empty(1, hidden_dim))
        self.w2_init = nn.Parameter(torch.empty(hidden_dim, dim))
        self.b2_init = nn.Parameter(torch.empty(1, dim))
        self.lr_multiplier_raw = nn.Parameter(torch.empty(()))
        self.gate = nn.Parameter(torch.empty(dim))
        self.gate_init = gate_init
        self.reset_parameters()

    def reset_parameters(self):
        for module in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(module.weight)
        nn.init.normal_(self.w1_init, std=self.dim**-0.5)
        nn.init.normal_(self.w2_init, std=self.hidden_dim**-0.5)
        nn.init.zeros_(self.b1_init)
        nn.init.zeros_(self.b2_init)
        nn.init.constant_(self.lr_multiplier_raw, math.log(math.expm1(1.0)))
        nn.init.constant_(self.gate, self.gate_init)

    def initial_state(self, batch_size, device):
        return FastWeights(
            *(
                p.float().to(device).unsqueeze(0).expand(batch_size, -1, -1)
                for p in (self.w1_init, self.b1_init, self.w2_init, self.b2_init)
            )
        )

    def forward(self, tokens, state, offsets, write_mask=None):
        # Projection follows the model dtype; fast learner math always uses fp32.
        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            q = apply_rope(q.float(), offsets, self.rope_theta)
            k = apply_rope(k.float(), offsets, self.rope_theta)
            if state is None:
                state = self.initial_state(tokens.shape[0], tokens.device)
            lr = self.inner_lr * F.softplus(self.lr_multiplier_raw.float())
            updated = sgd_step(k, v.float(), state, lr, write_mask)
            memory = fast_mlp(q, updated)  # update THEN read, as in RoboTTT
        return memory.to(tokens.dtype), updated


class TTTContext:
    """Explicit state for one sequence or a stable batch of environment streams."""

    def __init__(self, num_layers, batch_size, device):
        self.states = [None] * num_layers
        self.offsets = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.write_mask = torch.ones(batch_size, device=device, dtype=torch.bool)

    def fork(self, detach=False):
        other = TTTContext(len(self.states), len(self.offsets), self.offsets.device)
        other.states = [s.detach() if detach and s is not None else s for s in self.states]
        other.offsets = self.offsets.clone()
        other.write_mask = self.write_mask.clone()
        return other

    def detach(self):
        self.states = [s.detach() if s is not None else None for s in self.states]

    def apply(self, memory, tokens):
        idx = memory.layer_index
        out, self.states[idx] = memory(tokens, self.states[idx], self.offsets, self.write_mask)
        return tokens + torch.tanh(memory.gate).to(tokens.dtype) * out

    def advance(self, num_tokens):
        self.offsets = self.offsets + self.write_mask.long() * num_tokens
