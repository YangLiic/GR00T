# RoboTTT on this GR00T N1.7 baseline

This is an independent, PyTorch reference implementation of [RoboTTT](https://arxiv.org/abs/2607.15275),
not the authors' released code/checkpoint and not a reproduction of their reported success rates.
The public project page linked the paper, but did not expose a source-code release when checked.

## What is retained

The `GR00T_N1.7` branch snapshots the user's working pipeline, including its existing
LeRobot normalization, relative/absolute actions, BGR/RGB handling, horizon checks,
flow-time sampling overrides, Euler schedules, RTC path, checkpoint format and serving API.
`main` starts at the same baseline. Changes live on `RoboTTT` in a separate checkout.
`robottt_enabled=False` is the default: existing checkpoints and launch commands keep
their baseline path, without register tokens or fast-memory parameters.

The local baseline uses **Cosmos-Reason2/Qwen3-VL**. The paper appendix describes an
**Eagle** backbone. We preserve the working local backbone; this is a documented
backbone substitution, not an exact pretrained-model match.

## Implemented mechanisms and source mapping

| Paper | Implementation |
|---|---|
| Fig. 2, §3.1: TTT after attention in every DiT layer | `BasicTransformerBlock`: after attention/dropout, before its residual addition and FFN; existing alternating self/cross-attention retained |
| §3.1: 16 learned registers carrying VL context | Registers prepended to state and noised-action tokens at the DiT input; all positions propagate through all layers |
| Eqs. 1–2: K/V SGD, then query the updated MLP | `ttt_memory.py`: analytic differentiable SGD, two linear layers with exact GELU, update-before-read |
| Eq. 3, Fig. 3: channel-wise tanh gate | Each layer owns `gate`, initialized to 0.001 |
| Appendix A.1: learned initial state and inner step size | Learned W1/b1/W2/b2 initialization, positive learned scalar multiplier on base LR 0.1 |
| Appendix A.1: RoPE θ=10000 | Applied to Q/K over flattened temporal token positions; separate from action PE and flow time τ |
| §3.2: sequence action forcing | Existing Beta(1.5,1), s=0.999 sampler called independently for each sequence observation/chunk |
| §3.2: TBPTT | Backward immediately per segment, carry numerical fast weights, detach graph at boundaries; one outer optimizer step per sequence batch |
| §3.3: context versus imitation target | `robottt_loss_mask_column`: zero action loss does not disable memory writes; padded frames disable both |
| Deployment | Per-policy, per-layer, per-batch-member memory; `policy.reset()` clears it; supports `no_grad` and `inference_mode` |

Fast state is explicitly outside `state_dict` and outside the optimizer. A checkpoint
stores the learned initialization, projections, gate, registers and configuration;
it never stores the last episode's transient fast weights.

The paper does **not** publish the precise fast-MLP width, bias/normalization details,
or which denoising evaluation supplies the single committed write. These are explicit
reference choices here:

- One dense memory MLP per DiT layer, width 1536→768→1536 at the normal configuration,
  with biases; full-width Q/K/V projections. This adds about 9.44M parameters/layer,
  consistent with the paper's rounded ~10M/layer, but is not an author-confirmed width.
- Squared-error loss averaged over tokens and features, independently for each
  batch member. No extra normalization, decay, clipping or head split is silently added.
- All Euler/RTC denoising evaluations fork from the same preceding-observation state;
  each does update-then-read. Only the final evaluation's candidate is committed.
  Thus K denoising calls do not count as K physical observations. This transactional
  rule is an implementation assumption matching Fig. 2's one committed update per inference.
- Near-zero gates suppress the TTT branch. They do not guarantee exact baseline
  outputs when registers are enabled, since registers also participate in self-attention.

## Training directly on a task

Yes: initialize from the user's GR00T base/task checkpoint, add fresh memory modules,
and train on chronological demonstrations of the selected task. This **skips the
paper's 30K-step mixed-data sequence pretraining**. It cannot support claims about
the paper's 8K scaling results or one-shot capabilities without separate experiments.

The action loss still trains new memory through the original action network. No
robotics benchmark, full VLM training run, or success-rate measurement is launched
as part of implementing this code.

The existing finetune CLI accepts:

```bash
python gr00t/experiment/launch_finetune.py \
  --base-model-path /absolute/path/to/GR00T/checkpoint \
  --dataset-path /absolute/path/to/one/lerobot/task \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/RoboTwin/robotwin_aloha_config.py \
  --output-dir /absolute/path/to/new/robottt/run \
  --robottt-enabled --robottt-train-mode posttrain \
  --robottt-context-length 1024 --robottt-tbptt-steps 4 \
  --global-batch-size 1 --num-gpus 1 \
  --learning-rate 5e-5 --weight-decay 1e-5 \
  --max-steps 20000 --save-steps 1000 \
  --tune-llm --tune-visual --tune-projector --tune-diffusion-model
```

This is a template, not a claim that full-parameter training fits the development
L4. The default trainable-backbone flags otherwise retain ordinary finetune semantics.
Explicitly omit LLM/visual tuning for a cheaper action-head experiment, and report
that additional difference from the paper. `--robottt-train-mode memory` freezes
the original model and tunes only new modules on **task data**; it can be used as
optional task-local warm-up, not misreported as the paper's mixed-data pretraining.

For DDP use the existing torchrun launcher with matching `--num-gpus` and batch size.
This reference TBPTT explicitly supports single device/DDP; it rejects DeepSpeed,
FSDP, Apex and DataParallel instead of silently producing incorrect gradients.
Gradient checkpoint replay of mutable memory is disabled.

Context length counts physical timesteps; chunk stride defaults to H. Thus a
1024-step window with H=32 contains 32 observation/action-chunk positions, each
having 16 register + 1 state + H action tokens. TBPTT length counts these chunk
positions, not the physical timesteps or token positions. Windows never cross
episode boundaries; short episodes are padded, not concatenated into artificial histories.
The finite reference loader uses a rank-symmetric, epoch-seeded weighted **sequence**
sampler that preserves the existing dataset mixture weights. Sampling is with
replacement; episodes are only shuffled at the sequence level, never within a window.

Keep existing color conventions: offline RoboTwin data that stores BGR requires
`ROBOTWIN_BGR2RGB=1`. Live SAPIEN RGB observations require `ROBOTWIN_BGR2RGB=0`.
Do not change the channel convention on both client and server.

For DAgger-style data, an optional scalar parquet column can mark imitation targets
with `--robottt-loss-mask-column COLUMN`. Loss-masked frames still contain real
observations/actions and update memory. This is masking support, not a supplied
DAgger data-collection system. Human-video one-shot training additionally requires
correct paired data and missing-action handling; this reference loader does not
pretend that ordinary robot demonstrations provide those capabilities.

## Loading and deployment

The standard training pipeline upgrades baseline weights with strict validation:
only entirely new memory/register parameters may be missing. Missing original
weights, unexpected weights and partially missing trained memory are errors.
Saved checkpoints carry all `robottt_*` settings and load through the existing
`AutoModel` / `Gr00tPolicy` path. Never turn off `robottt_enabled` when loading a
trained RoboTTT checkpoint.

The development worktree shares the baseline venv: explicitly prepend this checkout
to `PYTHONPATH` so its editable install cannot resolve `gr00t` from the baseline.
The `examples/RoboTTT/run_task_posttrain.sh` and `run_server.sh` scripts do this.
For manual commands: `export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"` from
the RoboTTT checkout.

Use the existing PyTorch server, with one policy instance per independent stream
(a fixed batch of vectorized environments is allowed). Call `reset()` at each new
episode, environment/task switch or client reassignment. Current reset is for the
whole batch; independently terminating vectorized episodes need a per-row reset
adapter before use. Concurrent requests sharing one policy's memory are rejected
while a model call is active; sequential unrelated clients must also be isolated.
The existing baseline ONNX/TensorRT exporters are not a stateful TTT implementation;
export is deliberately rejected for RoboTTT until an explicit fast-state I/O contract exists.

## Validation and remaining experiments

`tests/robottt/` covers analytic SGD against autograd, meta-gradients, independent
batch memories, padding masks, frozen-policy gradient propagation, state-dict and
config roundtrips, baseline-weight preservation, both DiT variants, episode resets,
one committed write per observation, and two actual optimizer steps with segmented
backward. Run these with CPU tiny models; they do not download a VLM checkpoint.

Before reporting benchmark performance, select tasks/data, check train/eval RGB and
action contracts, test a real checkpoint on the target device, match observation
stride to execution cadence, and evaluate baseline versus RoboTTT under equal
data/interaction budgets. Short fully observable tasks may provide little benefit;
prefer occlusion, stage ambiguity or history-dependent tasks for the first comparison.
