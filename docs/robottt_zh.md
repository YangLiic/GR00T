# RoboTTT 开发说明

这是在你已经跑通的 GR00T N1.7 上实现的 **RoboTTT 参考版本**，代码与论文结果复现须分别看待。

## 分支与目录

- `main`、`GR00T_N1.7`：当前训练、推理、部署代码的基线快照，包含已有本地修改。
- `RoboTTT`：新增代码所在分支。
- 原目录 `/DRoboticsAILab/vepfs_20260622/licheng.yang/Isaac-GR00T` 保留为基线。
- 新开发目录 `/DRoboticsAILab/vepfs_20260622/licheng.yang/Isaac-GR00T-RoboTTT` 是独立 Git worktree。
- 旧 Git 元数据已完整备份到原目录的 `.git-history-20261021T043647Z/`，不上传 GitHub。
- 开发目录通过软链接复用原 `.venv`、`checkpoints`、`.hf_home`，不重复下载、不修改预训练权重。

## 已实现的路径

```text
每个时间步：视觉语言特征 + 状态 + 加噪动作 + 16 个寄存器
    → 原 DiT 注意力
    → 各层独立的 TTT：K/V 误差更新两层 GELU MLP，再用 Q 读取
    → tanh 门控加回注意力输出
    → 原 FFN、残差与后续层
    → 原归一化、投影、动作解码器
```

训练数据按同一 episode 的时间顺序读取，动作 chunk 独立采样噪声。
TBPTT 每个片段立即反传，再保留 fast weight 数值、截断计算图；不是先积累整个长序列的图再统一 backward。
Padding 不写记忆、不计算动作损失；真实上下文即使没有动作损失，也可以写记忆。

模型保存 QKV、门控、初始化 W0、寄存器等长期参数；运行中的记忆不写入 checkpoint。
部署时，使用新分支的 PyTorch 推理代码，episode 开始调用 `policy.reset()`。
共享 venv 的 editable install 仍指向基线，因此手动执行时必须从新目录设置 `PYTHONPATH="$PWD:$PYTHONPATH"`。
`examples/RoboTTT/run_task_posttrain.sh` 和 `run_server.sh` 已自动处理，避免误导入旧模型。
当前按整个环境 batch 重置，独立结束的向量环境需后续接入逐行 reset。

## 能否跳过第一阶段

可以直接从已有 GR00T checkpoint 开始，在特定任务的序列数据上训练新增记忆与策略。
入口仍然是 `gr00t/experiment/launch_finetune.py`，增加 `--robottt-enabled`。
完整模板见 `examples/RoboTTT/run_task_posttrain.sh`；任务、数据、输出路径由环境变量显式提供。

- `--robottt-train-mode posttrain`：遵循原来的 tune 参数；论文全参训练需要显式打开 LLM、视觉、projector 与 diffusion model。
- `--robottt-train-mode memory`：只训练新增模块，可用于任务内短暂热身，不能称为完成论文的大规模预训练。
- `--robottt-context-length 1024`：以物理时间步计的上下文上限。
- `--robottt-chunk-stride 0`：默认按动作 horizon 采样；若部署每次执行更短前缀，需显式匹配采样间隔。
- `--robottt-tbptt-steps 4`：每 4 次观测/chunk 构成一个反传片段。

跳过论文第一阶段后，长程记忆能力需要在任务数据中学出来，不能预先承诺 8K 效果或 one-shot 泛化。
初步建议先选择确实依赖历史的任务，再与同数据、同预算的 GR00T 基线比较。

## 必须记录的复现差异

1. 本地稳定主干是 Cosmos/Qwen3-VL，论文附录描述的是 Eagle；不为对齐名称而替换已跑通主干。
2. 作者未披露 fast MLP 的确切宽度。这里选 1536→768→1536，参数量约 9.44M/层，吻合其约 10M 的数量级，但不是作者确认的配置。
3. 作者未完整说明多次去噪中的写入安排。这里各次去噪从同一份上次观测记忆出发，只提交最后一次的候选更新，避免把一次观测当成多次历史。
4. 当前支持单设备和 DDP 参考训练，明确拒绝尚未验证的 DeepSpeed/FSDP；完整模型的显存预算尚未实测。
5. Stateful TTT 的 ONNX/TensorRT 导出尚未实现，不能直接复用原来的无状态引擎导出器。
6. 本次没有启动大模型训练或 RoboTwin/RoboMME/RMBench 闭环评测；这些任务与适配器待后续确定。

英文架构、训练和部署说明见 [robottt.md](robottt.md)。
