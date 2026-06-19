# Agent 设计说明与评审

本文档整理当前仓库中 Agent 相关设计，并给出建议的完善方向。范围包括已经接入主评测框架的 `generate_until_agentic`，以及当前工作区新增但尚未完成的 VizDoom/Agent 草案文件。

## 当前状态

Agent 能力目前分成两条线：

1. 已落地的 agentic evaluation：通过 `output_type: generate_until_agentic` 复用现有 evaluator，让模型多轮输出 `<tool_call>` 或 `<submit>`，由 task 侧 `doc_to_text` 执行工具或模拟器并返回下一轮 prompt。
2. 新增的 Agent/VizDoom 草案：在 `lmms_eval/models/agent/` 和 `lmms_eval/tasks/vizdoom/` 下尝试引入 `AgentModel`、`ActionParser`、`ObservationParser` 和 `AgentLoop` 等抽象，但目前基本是占位代码，还不能被框架加载或运行。

因此，现阶段应把 `generate_until_agentic` 视为可用基础设施，把 `models/agent` 与 `tasks/vizdoom` 视为下一步要对齐到该基础设施的设计草案。

## 设计目标

- 让评测任务能够表达“观察环境 -> 模型决策 -> 执行动作 -> 更新环境 -> 再观察”的闭环。
- 保持 lmms-eval 的 YAML/task/model registry 使用方式，不为 Agent 任务另起一套入口。
- 支持文本工具任务和多模态交互任务，VizDoom 属于后者。
- 保证 simulator/env 可复现，评测结果可审计，trace 可落盘分析。
- 让模型适配、环境逻辑、动作解析、指标计算各自保持清晰边界。

## 已有 Agentic Evaluation 架构

### Task 配置入口

Agentic task 使用普通 task YAML：

```yaml
output_type: generate_until_agentic
doc_to_visual: !function utils.xxx_doc_to_visual
doc_to_text: !function utils.xxx_doc_to_text
generation_kwargs:
  max_new_tokens: 256
  temperature: 0
  max_agentic_steps: 10
process_results: !function utils.xxx_process_results
```

`ConfigurableTask.construct_requests()` 和 `ConfigurableMessagesTask.construct_requests()` 会把每个样本构造成 7 元组：

```python
(
    ctx,
    generation_kwargs,
    doc_to_visual,
    doc_to_text,
    doc_id,
    task_name,
    split,
)
```

这个 7 元组是当前 `generate_until_agentic` 的实际协议。

### Evaluator Loop

`lmms_eval/evaluator.py::_run_generate_until_agentic()` 是主循环。每个样本执行：

1. 读取 `max_agentic_steps`，并从模型 generation kwargs 中移除。
2. 使用当前 context 构造一次普通 `generate_until` 请求。
3. 调用模型得到本轮输出。
4. 调用 task 侧 `doc_to_text(doc, previous_output, round_idx, previous_round_info)`。
5. task 解析模型输出，执行工具或环境动作，并返回下一轮输入、终止信号和更新后的 state。
6. 若终止则停止；若未终止则更新 `current_context` 和可选 visuals，进入下一轮。
7. 若达到步数上限仍未正常提交，生成 fallback JSON。

`doc_to_text` 可以返回两种形式：

```python
# 简化形式：只返回下一轮 prompt
str

# 完整形式
(
    visuals,             # 下一轮视觉输入；None 表示不更新
    next_context,        # 下一轮文本输入；None 表示不更新
    terminal_signal,     # 是否结束 loop
    updated_outputs,     # 用于覆盖 previous_output / final response
    next_round_info,     # task 侧状态与 trace 信息
)
```

### 已有参考任务

当前可参考的实现：

- `lmms_eval/tasks/vending_bench2/`：纯 Python vending simulator，模型调用工具调价、补货、模拟天数，最终提交结果。
- `lmms_eval/tasks/tau2_bench/`：telecom 状态修复任务，支持 `<tool_call>`、`<submit>`，并额外兼容一些 JSON-like 输出。

这两个任务说明了当前推荐模式：环境状态放在 `previous_round_info["state"]`，工具执行发生在 task utils 中，最终结果由 `process_results` 解析 JSON payload 后计算指标。

## 建议的完善后设计

### 总体分层

推荐把 Agent 任务拆成五层：

```text
YAML task
  -> evaluator generate_until_agentic loop
    -> AgentLoop / task utils
      -> ObservationParser
      -> model.generate_until
      -> ActionParser
      -> Environment / simulator
      -> Metrics / trace payload
```

各层职责如下：

| 层 | 职责 | 不应负责 |
| --- | --- | --- |
| YAML task | 声明 dataset、output_type、callbacks、generation kwargs、metrics | 环境运行细节 |
| evaluator | 调度模型多轮生成、缓存、步数预算、trace mode | 理解具体游戏规则或工具语义 |
| AgentLoop/task utils | 管理 episode state、调用 parser、step env、构造下一轮输入 | 模型加载和底层推理 |
| ObservationParser | 把 env observation 转成模型可消费的文本/图像/视频/audio 内容 | 执行动作 |
| ActionParser | 把模型输出转成结构化 action/submit/error | 修改环境状态 |
| Environment/simulator | 执行动作、返回 observation/reward/done/info | 解析自然语言模型输出 |
| Metrics | 解析 final payload，计算 success、progress、step validity、termination quality | 改变 episode 状态 |

### 推荐接口

当前 `proto` 文件可以发展成轻量协议和数据结构，而不是模型专属基类。

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

@dataclass
class AgentAction:
    kind: Literal["action", "submit", "invalid"]
    name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    error: str | None = None

@dataclass
class AgentObservation:
    text: str
    visuals: list[Any] = field(default_factory=list)
    state_summary: dict[str, Any] = field(default_factory=dict)

class ObservationParser(Protocol):
    def parse(self, observation: Any, state: dict[str, Any]) -> AgentObservation:
        ...

class ActionParser(Protocol):
    def parse(self, model_output: str) -> AgentAction:
        ...
```

`AgentModel` 不一定需要独立存在。lmms-eval 已经有模型 registry 和 `generate_until` 接口，Agent 任务最好优先复用现有模型接口。若确实需要 AgentModel，也应只是适配器：

```python
class AgentModel(Protocol):
    def generate(self, observation: AgentObservation, generation_kwargs: dict[str, Any]) -> str:
        ...
```

### VizDoom 接入方式

VizDoom 不建议使用单独的 `task_type: agent` 入口。更稳妥的方式是仍然定义标准 task：

```yaml
task: vizdoom_single_player
output_type: generate_until_agentic
doc_to_visual: !function utils.vizdoom_doc_to_visual
doc_to_text: !function utils.vizdoom_doc_to_text
process_results: !function utils.vizdoom_process_results
generation_kwargs:
  max_new_tokens: 128
  temperature: 0
  max_agentic_steps: 200
```

`vizdoom_doc_to_text` 内部负责创建或恢复 episode state：

1. 首轮构造初始 observation prompt。
2. 后续轮次从 `previous_output[-1]` 解析 action。
3. 调用 VizDoom env step。
4. 把新 frame/game variables/reward/done 转为下一轮 `visuals` 和 `next_context`。
5. done 或 submit 时返回 terminal payload。

建议 final payload 至少包含：

```json
{
  "success": true,
  "score": 12.0,
  "episode_length": 83,
  "terminal_reason": "goal_reached",
  "valid_actions": 80,
  "invalid_actions": 3,
  "state": {"health": 80, "ammo": 4},
  "trace": []
}
```

对 VizDoom 这类环境，必须显式定义：

- seed、scenario、map、difficulty。
- action space 和非法动作策略。
- frame skip / action repeat。
- episode timeout 与 `max_agentic_steps` 的关系。
- observation 编码方式：单帧、短视频、frame stack，还是文本化 game variables。
- 评测指标：win rate、score、survival time、damage dealt、invalid action rate 等。

## 当前设计问题

### P0：新增 Agent/VizDoom 草案不可运行

当前新增文件存在基础语法和引用问题：

- `lmms_eval/models/agent/proto/model.py` 使用了未定义的 `@abstract`，且 `def generate(self, ...):` 不是有效接口定义。
- `lmms_eval/models/agent/llava_onevision2/model.py` 缺少冒号，也没有 import `AgentModel`。
- `register_action_parser`、`register_observation_parser` 没有对应 registry 定义或 import。
- `LLaVAOneVision2_ViZDoom_ObservationParser` 继承了 `ActionParser`，应继承 `ObservationParser`。
- `SinglePlayerAgentLoop`、`TwoPlayerCompactAgentLoop` 继承的 `AgentLoop` 未定义。
- `lmms_eval/tasks/vizdoom/vizdoom.yaml` 只有 `task_type: agent`，不能被现有 lmms-eval task loader 当作完整任务运行。

这些问题说明当前草案还不能作为设计契约，只能作为方向草图。

### P0：草案入口与现有框架协议不一致

主框架已经用 `output_type: generate_until_agentic` 表达 Agent 任务，而 VizDoom 草案用了新的 `task_type: agent`。如果继续另起入口，会导致 task loader、evaluator、cache、metrics、CLI 参数都要重复接一遍。

建议统一到 `generate_until_agentic`，把 VizDoom loop 实现为 task utils 或可复用 `AgentLoop`，而不是新建并行 task type。

### P1：终止语义和 fallback 语义混在一起

当前 `_run_generate_until_agentic()` 在 loop 结束后，如果 `previous_round_info` 存在且 `final_response` 不是 JSON，就会生成 `"error": "max_agentic_steps_reached"` fallback。这个逻辑没有区分“正常 terminal 但 final response 非 JSON”和“真的达到 max steps”。

更合理的设计是显式记录 `terminal_reached` 和 `exhausted_steps`：

- terminal 后如果 task 没返回 JSON，应标记为 `invalid_terminal_payload`，不应标记为 max steps。
- 只有没有 terminal 且循环耗尽时，才返回 `max_agentic_steps_reached`。

### P1：`doc_to_text` 职责过重

当前 `doc_to_text` 同时负责：

- 首轮 prompt 构造。
- 模型输出解析。
- 工具/env 执行。
- 状态更新。
- 下一轮 prompt 构造。
- final payload 构造。

短期可以接受，但随着 VizDoom 这种复杂环境加入，会很快膨胀。建议把内部逻辑拆成 `ActionParser`、`ObservationParser`、`AgentLoop.step()` 和 `build_prompt()`，外层仍由 `doc_to_text` 适配 evaluator。

### P1：parser 维度耦合过紧

当前草案路径是 `models/agent/llava_onevision2/action_parsers/vizdoom.py`，这把“模型格式”和“任务环境”绑在一个类里。更好的拆法是：

- model-specific：处理某个模型的输入输出格式偏好。
- env-specific：处理 VizDoom action space 和 observation。
- task config：组合 model format parser 与 env parser。

如果 parser 确实只服务某个模型和某个环境，也应在命名和文档里明确这是 adapter，而不是通用 parser。

### P1：状态结构没有 schema

`previous_round_info` 是自由 dict，task 可以随意写入 `state`、`tool_calls`、`invalid_steps` 等字段。优点是灵活，缺点是没有类型约束，跨任务复用困难，trace 和 metrics 容易漂移。

建议引入最小公共字段：

```python
{
    "state": dict,
    "step": int,
    "valid_steps": int,
    "invalid_steps": int,
    "last_action": dict | None,
    "last_result": dict | None,
    "terminal_reason": str | None
}
```

任务可以扩展字段，但公共 metrics 和 trace 只依赖这些字段。

### P2：输出协议过于依赖 prompt 约束

`vending_bench2` 和 `tau2_bench` 通过 prompt 要求模型输出 `<tool_call>` 或 `<submit>`。这容易受模型格式影响。`tau2_bench` 已经额外兼容 JSON-like 输出，说明单一 tag 机制不够稳。

建议把解析策略抽象为 `ActionParser`，并支持：

- strict tag parser：用于可控模型和回归测试。
- JSON/function-call parser：用于 OpenAI-style 工具调用模型。
- tolerant parser：用于 benchmark 兼容，但必须记录 parsing repair 次数。

### P2：多工具调用语义不明确

prompt 说“exactly one tool”，但实现会执行所有解析出的 tool calls，并只把最后一次 tool result 写回 prompt。模型如果一轮输出多个工具，评测到底算一次复合动作还是多次动作不清晰。

建议选择一个明确策略：

- strict：一轮只接受一个 tool call，多余的算 invalid。
- batch：允许多个 tool call，但每个 tool call 都算一步或子步，并完整记录全部结果。

VizDoom 这类动作环境建议使用 strict 策略。

### P2：多模态 media 类型推断过于粗糙

chat model 路径里根据 Python 类型推断 media：

- `dict` -> audio
- `str` -> video
- 其他 -> image

这对 VizDoom 不够可靠。建议定义显式 media object，例如：

```python
{"type": "image", "data": frame}
{"type": "video", "url": path}
{"type": "audio", "url": path}
```

并让 `doc_to_visual` 或 `ObservationParser` 返回明确类型。

### P2：性能上是逐样本逐轮串行

当前 evaluator 对每个 request 单独跑完整 loop。Agent 任务本身通常较慢，短期问题不大；但大规模 benchmark 会很慢。后续可以把同一 round 的 active requests batch 到一次 `generate_until` 调用，再分别 step 各自环境。

### P2：公共指标重复实现

`vending_bench2` 和 `tau2_bench` 都重复实现了 step validity、state progress、termination quality、trace quality。建议提供公共 helper，让任务只实现 domain-specific progress：

```python
compute_trace_step_validity(payload)
compute_termination_quality(payload)
compute_trace_quality(step_validity, state_progress, termination_quality)
```

## 建议推进顺序

1. 先修正新增 Agent/VizDoom 文件的语法、import、继承和 registry 缺失问题，或者在实现前删除 decorators，避免不可导入代码进入主分支。
2. 明确 VizDoom 也走 `output_type: generate_until_agentic`，不要引入独立 `task_type: agent`。
3. 在 `tasks/vizdoom/utils.py` 里先实现一个最小可运行单人 episode loop，再抽 `AgentLoop`。
4. 把 `ActionParser` 和 `ObservationParser` 做成任务内部可组合组件，跑通后再考虑迁移到公共 proto。
5. 修正 evaluator 的 terminal/max-steps fallback 语义，并补测试覆盖。
6. 引入公共 trace schema 与公共 trace metric helper，减少每个 Agent task 重复代码。
7. 最后再做 batched active-loop 调度和更完整的 full trace 控制。

## 推荐验收标准

一个 Agent task 合入前至少满足：

- YAML 可被普通 `python -m lmms_eval --tasks <task>` 发现和运行。
- `--limit 1 --batch_size 1` 能完成一次 episode。
- 固定 seed 下结果可复现。
- 非法 action、无 action、重复 action、max steps、正常 submit 都有测试。
- final payload 是稳定 JSON schema。
- `process_results` 不依赖未记录的外部状态。
- full trace 不默认写入过大的 frame/raw observation。
