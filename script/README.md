# One-shot Evaluation Scripts

`scripts/` 目录下提供了两个串行执行的“一次性跑完整套任务”脚本：

- `eval_all.sh`：普通任务版本
- `eval_all_cot.sh`：Visual CoT / CoT 任务版本

这两个脚本的目标很简单：给定一个 `lmms_eval` 已支持的模型名和对应 `model_args`，按固定任务列表逐个执行评测，并在全部完成后自动汇总结果。

## 推荐用法

建议始终在仓库根目录运行，而不是先 `cd scripts` 再执行。这样输出会统一写到仓库根目录下的 `logs/`。

普通版本：

```bash
bash scripts/eval_all.sh \
  --model qwen2_5_vl \
  --model_args "pretrained=Qwen/Qwen2.5-VL-3B-Instruct"
```

CoT 版本：

```bash
bash scripts/eval_all_cot.sh \
  --model bagel_visual_cot \
  --model_args "pretrained=ByteDance-Seed/BAGEL-7B-MoT,save_intermediate=true"
```

## 参数说明

两个脚本都只接收两个参数：

- `--model`：`lmms_eval` 中注册的模型名，例如 `qwen2_5_vl`、`bagel_visual_cot`
- `--model_args`：传给 `lmms_eval` 的模型参数字符串，例如 `pretrained=...`、`device_map=auto`、`save_intermediate=true`

对应到底层执行命令：

```bash
uv run python -m lmms_eval \
  --model "$MODEL" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK" \
  --batch_size 1 \
  --log_samples \
  --output_path "${OUTPUT_BASE}/${TASK}"
```

## 运行前提

- 已在当前环境安装本仓库依赖
- 命令行中可直接使用 `uv`
- 目标模型已经在 `lmms_eval` 中实现，并且 `--model_args` 填写正确
- 对应 benchmark 所需的数据集、权限或 API key 已按仓库主 README 配置完成

如果你的环境不是通过 `uv` 管理，这两个脚本里的 `uv run python` 需要改成你自己的 Python 启动方式。

## 脚本行为

两个脚本都会：

- 按脚本内固定的任务列表串行执行
- 强制设置单机单卡分布式环境变量，避免误连远程节点
- 固定使用 `--batch_size 1`
- 开启 `--log_samples`
- 任一任务失败后立即退出，因为脚本使用了 `set -e`
- 在所有任务完成后扫描结果文件并生成汇总

脚本会覆盖这些环境变量：

```bash
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29314
```

## 任务列表

### `eval_all.sh`

包含以下 11 个普通任务：

```text
auxsolidmath_easy
chartqa100
geometry3k
babyvision
illusionbench_arshia_test
mmsi
phyx_simple
realunify
uni_mmmu
vsp
VisualPuzzles
```

### `eval_all_cot.sh`

包含以下 11 个 CoT / Visual CoT 任务：

```text
auxsolidmath_easy_visual_cot
chartqa100_visual_cot
geometry3k_visual_cot
babyvision_cot
illusionbench_arshia__visual_cot_split
mmsi_cot
phyx_cot
realunify_cot
uni_mmmu_cot
vsp_cot
VisualPuzzles_visual_cot
```

## 输出结构

默认输出目录为：

```text
logs/<model>/
```

每个任务都会单独写入自己的子目录：

```text
logs/<model>/
├── <task_1>/
├── <task_2>/
├── ...
└── summary.json
```

`lmms_eval` 自身通常还会在任务目录下再创建时间戳或运行实例子目录，所以最终的 `results*.json` 往往位于更深一层。脚本最后会自动扫描这些结果文件，终端打印汇总表，并额外写出：

```text
logs/<model>/summary.json
```

`summary.json` 只保留各任务的主要指标，自动过滤 `alias` 和 `stderr` 字段。

## 常见注意事项

- 建议从仓库根目录执行：`bash scripts/eval_all.sh ...`
- 如果从其他目录启动，`./logs` 会相对于当前工作目录创建
- 脚本当前不支持断点续跑，某个任务失败后需要重新执行整个脚本，或手动改 `TASKS` 数组
- CoT 模型通常对 `model_args` 更敏感，尤其是中间结果保存、图像输出等参数，建议先单任务验证模型能否正常跑通

## 建议流程

为了避免整套任务跑到一半才发现模型参数有问题，建议先做两步检查：

1. 先用同一个 `--model` 和 `--model_args` 单独跑一个任务
2. 确认输出目录、样本日志和最终指标都正常后，再跑整套脚本
