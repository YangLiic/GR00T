# RoboTTT reference validation — 2026-10-21

Baseline snapshot: `1f80a69` (`main` and `GR00T_N1.7`).
Tests used the existing Python 3.12 / PyTorch 2.9.0+cu128 environment, with CUDA
hidden and isolated pytest/ruff/pre-commit tools; the project's environment was
not upgraded. No model weights were downloaded and no benchmark training was run.

| Check | Observed result |
|---|---|
| `pytest tests/robottt tests/gr00t/model/test_action_head.py -q -p no:cacheprovider` | **27 passed**, final run 82.49 s; inherited dependency/deprecation warnings retained |
| `torchrun --standalone --nproc_per_node=2 tests/robottt/run_ddp_smoke.py` | **Both ranks completed 2 optimizer steps**, trained parameters equal across ranks |
| `launch_finetune.py --help` | Exit 0, new RoboTTT arguments available |
| Ruff on changed Python files | Passed after formatting |
| `bash -n examples/RoboTTT/run_task_posttrain.sh` | Passed |
| Full-repository pre-commit audit, read-only equivalents of pinned Ruff hooks | Existing baseline lint/format failures remain; cross-platform manifest alignment passed |

The full-repository hook failures were observed on the untouched baseline before
RoboTTT changes. We did not autoformat or refactor unrelated inherited files to
make an experimental branch appear globally clean. The new/modified Python files
are checked separately with Ruff 0.12.7.

Coverage includes:

- Analytic fast-weight SGD and meta-gradients compared numerically to autograd.
- No memory sharing across batch members; padding does not write.
- Action gradients through frozen original networks reach memory parameters.
- Initialization from baseline preserves every original state-dict tensor.
- Actual `AutoModel.from_pretrained` upgrade, `save_pretrained` and strict reload
  with a tiny fake backbone; the backbone substitution avoids a real VLM download.
- Both DiT variants in inference mode, deterministic reset, one committed history
  update despite multiple denoising calls.
- Episode-bounded windows and padded tails; weighted sampling reproducible by epoch.
- Segmented backward followed by one outer optimizer update; carried values are
  detached at segment boundaries, and transient memory is excluded from checkpoints.

The DDP smoke caught a real integration issue: PyTorch unused-parameter discovery
could not traverse `BatchFeature` as the top-level sequence result. Returning a
plain dictionary fixed the issue and was verified on both ranks. This is why a
single-device gradient test alone is not sufficient validation of this trainer.

Still unmeasured: full-size model GPU memory/latency, real task convergence,
closed-loop success, cross-task transfer, 8K contexts and the paper's benchmark
numbers. DeepSpeed/FSDP and stateful ONNX/TensorRT export are explicitly unsupported
by this reference path, rather than reported as tested.

Remote logs: `Isaac-GR00T-RoboTTT/logs/robottt-final-tests.log`,
`robottt-ddp.log`, `robottt-cli-help.log`, `robottt-lint.log`,
`robottt-precommit.log`. These logs are intentionally not committed.
