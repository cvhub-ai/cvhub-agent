# CVHub Agent — Workflow & Orchestration 设计归档

> **项目**：CVHub / `cvhub-agent`  
> **主题**：Phase 6 — Workflow & Orchestration  
> **整理日期**：2026-11-02  
> **用途**：保存本轮设计讨论、已实现方案及后续演进决定，供未来编写 README、架构文档及开发说明时引用。  
> **状态说明**：本文严格区分 **已实现/已有测试**、**已确定但暂缓**、**未来可选增强**。本文依据讨论与展示的代码整理，不代表已在当前代码仓库中重新运行并验证所有测试。

---

## 1. 背景与目标

CVHub 是一个通用 AI Agent 平台，集成 LLM、RAG、Computer Vision 工具及其他外部微服务。Agent 负责理解用户目标、规划任务、选择可用工具、组织执行、审核结果，而不直接承担 OCR、RAG 等领域服务内部的 Pipeline 实现。

Phase 6 的目标是使用 **LangGraph** 构建可控、可测试的任务级编排能力，在保留 LangChain Agent 动态工具选择能力的同时，使执行顺序、状态转换、失败终止和重试机制由显式 Workflow 管理。

主要原则：

1. **通用优先**：不要把特定客户的业务场景硬编码进通用 Agent 内核。
2. **职责分离**：Workflow 编排任务；Agent 自主完成步骤；MCP/工具层连接外部服务；OCR/RAG 服务管理各自内部的工作流程。
3. **受约束的自主性**：LLM 可以生成计划、选择工具，但关键控制流、失败终止和权限规则由代码约束。
4. **复用已有能力**：Executor 调用已经实现的 LangChain Agent，不重新开发 Tool Calling 框架。
5. **可测试性**：Unit Tests 使用 Fake/Mock Nodes，不访问 DeepSeek；Integration Tests 独立验证真实 LLM 和工具集成。
6. **渐进扩展**：先完成通用 Workflow，之后再引入用户定义的流程注册、持久化和精细工具绑定。

## 2. 核心概念与模式

### 2.1 四个容易混淆的概念

| 概念 | 含义 | 在 CVHub 中的定位 |
|---|---|---|
| **Direct Agent** | 直接向已有 LangChain Agent 提交任务，由 Agent 决定工具调用 | 不经过 Phase 6 的任务级 Planner → Executor → Reviewer Graph |
| **Dynamic Plan Workflow** | LLM 根据当前 Goal 生成一次性 Plan，再交给 Workflow 执行 | **当前已实现的主要模式** |
| **Predefined Plan Workflow** | 调用方提供预定义步骤，跳过动态 Planner，复用 Executor / Reviewer | **设计已确定，实现暂缓** |
| **Hybrid Workflow** | 固定的编排/控制结构与 Agent 的动态决策结合 | **当前已实现的编排思想**；不是独立于前两种 Plan 来源的另一套引擎 |

**重要区别**：

- **固定 Plan**：确定“有哪些步骤、按什么顺序执行”，但如果 Executor 仍调用 Agent，Agent 可以在步骤内部自主选择工具。
- **严格 Deterministic Workflow**：控制流与步骤执行器/工具绑定均由代码或配置确定，不依赖 LLM 自主选择工具；外部服务返回值仍不保证完全确定。
- 因此，**预定义文本 Plan + 现有 Agent Executor** 是一种 **固定计划的 Hybrid Workflow**，不能简单等同于严格确定性执行。

### 2.2 目前的任务执行主链路

```text
User Goal
    |
    v
PlannerNode  [LLM: generate ordered Plan]
    |
    v
ExecutorNode  <----------------------+
    |                                 |
    | LangChain Agent: Tool Calling   |
    |                                 |
    +-- step succeeded, more steps ---+  [continue, step_index + 1]
    |
    +-- retryable error --------------+  [retry same step, bounded]
    |
    +-- unrecoverable/explicit failure ----> END [failed]
    |
    v
ReviewerNode  [LLM: evaluate goal satisfaction]
    |
    +-- approved ----> Finish ----> END [completed]
    |
    +-- rejected ----> Reject ----> END [failed]
    |
    +-- reviewer error ------------> END [failed]
```

这里的“成功”是 **Reviewer 根据收到的执行内容做出认可**，不意味着外部事实已被独立验证。

## 3. 已实现：LangGraph 架构与模块职责

核心目录（讨论和代码中的路径）：

```text
src/cvhub_agent/workflows/
├── state.py
├── planner.py
├── executor.py
├── reviewer.py
└── workflow.py
```

### 3.1 `WorkflowState`：执行过程的统一状态

当前选择使用 Python **`dataclass`**，而不是 `TypedDict`；这是明确的设计选择。Node 内可使用 `state.goal`、`state.plan` 等属性读取状态，Node 返回 `dict` 形式的**部分更新**，由 LangGraph 合并。

```python
from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Literal

@dataclass
class StepResult:
    step: str
    output: str

@dataclass
class ReviewResult:
    approved: bool
    feedback: str

@dataclass
class WorkflowState:
    goal: str
    plan: list[str] = field(default_factory=list)
    step_index: int = 0
    results: Annotated[list[StepResult], add] = field(default_factory=list)
    status: Literal[
        "pending", "running", "retrying", "completed", "failed"
    ] = "pending"
    error: str | None = None
    review: ReviewResult | None = None
    step_retry_count: int = 0
```

字段说明：

| 字段 | 作用 |
|---|---|
| `goal` | 用户的原始目标；Planner、Executor、Reviewer 都需要了解 |
| `plan` | 有序步骤列表；目前为 `list[str]`，而不是复杂流程 Schema |
| `step_index` | 下一步待执行步骤的下标；成功执行一步后加 1 |
| `results` | 已完成步骤的输出；使用 `Annotated[..., operator.add]` 将新结果追加到原有列表 |
| `status` | 工作流生命周期：`pending`、`running`、`retrying`、`completed`、`failed` |
| `error` | 错误摘要或审核拒绝原因 |
| `review` | 最终 Reviewer 输出：`approved`、`feedback` |
| `step_retry_count` | **当前步骤**的重试次数，成功后清零 |

**注意**：`results` 的 reducer 是追加语义，所以一次重试失败时不得写入虚假的 `StepResult`；只能在成功时提交结果。

### 3.2 `PlannerNode`：动态计划生成

- 输入：`WorkflowState.goal` 和当前提供的工具清单/工具描述。
- 使用 `BaseChatModel.with_structured_output(PlanOutput, method="json_mode")`。
- `PlanOutput` 当前定义 `steps: list[str]`，限制为 **1～5 步**。
- Prompt 要求使用已有能力，不编造工具，不在 Planner 阶段执行工具；缺少能力时应在步骤中说明限制。
- 返回部分 State 更新：

```python
{
    "plan": response.steps,
    "step_index": 0,
    "status": "running",
    "error": None,
}
```

目前 Planner 对**本次 Goal**即时规划，不负责将计划注册成持久模板。

### 3.3 `ExecutorNode`：执行当前一步

- 由 `ExecutorNode(agent: Runnable)` 复用已创建的 LangChain Agent。
- 使用 `state.step_index` 找到当前 `plan` 中的一条指令。
- 将 `goal`、当前步骤、之前的 `StepResult` 作为上下文传给 Agent。
- Agent 可以使用现有 Tool Calling，工具由工具加载层提供。
- 校验 Agent 返回的最后消息是完成的、非空文本 `AIMessage`，不接受未完成的 Tool Calls。
- 成功时只返回当前步骤的增量结果和递增后的索引：

```python
{
    "results": [StepResult(step=step, output=output)],
    "step_index": state.step_index + 1,
}
```

Executor **不应**自行生成下一轮 Plan，也不应该在内部复制 OCR/RAG 服务的处理 Pipeline。

### 3.4 `ReviewerNode`：完成情况审核

- 输入：原始 Goal 与所有已完成步骤的结果。
- 只有计划执行完成、结果数量与步骤数量一致、结果与计划顺序对应时，才能进入审核。
- 使用 LLM 结构化输出 `ReviewOutput(approved: bool, feedback: str)`。
- Prompt 要求拒绝不完整、矛盾、明确报告失败的执行结果；不允许把结果中的内容当成指令，不应宣称做了独立外部验证。
- 返回：`{"review": ReviewResult(approved=..., feedback=...)}`。
- 审核通过进入 Finish；拒绝进入 Reject；Reviewer 自身异常进入 Failed 终态。

### 3.5 `TaskWorkflow`：编排与路由

`TaskWorkflow` 负责实例化 `StateGraph(WorkflowState)`、注册 Nodes、连接 Edges、管理执行循环，以及包装运行入口。当前固定的 Nodes：

- `planner`：外层包装 `_run_planner()`，负责捕获 Planner 异常与检查空 Plan。
- `executor`：外层包装 `_run_executor()`，负责 Retry、显式失败以及结果有效性检查。
- `reviewer`：外层包装 `_run_reviewer()`，负责捕获异常与检查 `ReviewResult` 类型。
- `finish`：设置 `status="completed"`。
- `reject`：设置 `status="failed"`，并保留审核反馈作为错误原因。

推荐的完整分支映射（已在讨论中确认为必须存在）：

```text
START -> planner

planner:
  execute -> executor
  failed  -> END

executor:
  continue -> executor
  retry    -> executor
  review   -> reviewer
  failed   -> END

reviewer:
  approved -> finish -> END
  rejected -> reject -> END
  failed   -> END
```

**已发现并修正的关键设计问题**：

1. **不能**固定使用 `planner -> executor` 普通 Edge，否则 Planner 报错后也会进入 Executor。改为 Planner 的 Conditional Edge。
2. Executor 结束全部步骤后必须进入 **`reviewer`**，不能直接进入 `finish`。早期曾因为错误地设置 `complete -> finish` 导致 Reviewer 没有运行。
3. 需要显式区分 `continue` 与 `retry` 两个路由标签，尽管二者均指向 Executor；一个进入**下一步**，另一个重新执行**当前步**。
4. Reviewer 异常已经被包装为 `status="failed"` 时，`_route_after_review()` 不能再抛 `ValueError`；必须具有 `failed -> END` 分支。
5. Planner、Executor、Reviewer 产生的状态只在合适时流向相应 Node；失败时应停止，而不是继续执行。

## 4. 已实现：Workflow Retry 与 Failure Handling

### 4.1 两层 Retry 的区别

| 层级 | 负责什么 | 当前策略 |
|---|---|---|
| **Tool Retry** | Agent 内部单次工具调用的临时故障，例如工具请求超时 | 由已有的 Tool/Middleware 机制处理 |
| **Workflow Step Retry** | 当前整个 Executor Step 抛出可重试异常 | 由 `TaskWorkflow` 包装与 Conditional Routing 处理 |

**默认 `max_step_retries=0`**，即**不自动重跑整个 Step**。因为 Agent 的单个 Step 可能已经触发多个工具或有副作用操作；无法确认幂等性时重试会有重复写入、重复通知等风险。

### 4.2 当前 Retry 规则

构造方式：

```python
TaskWorkflow(
    planner=planner,
    executor=executor,
    reviewer=reviewer,
    max_step_retries=0,
)
```

- `max_step_retries` 是**最多重试次数**，不包括第一次尝试。
- 例如 `max_step_retries=2`：首次尝试 + 最多两次重试，共最多三次执行。
- 当前只对 Executor 抛出的 **`TimeoutError`、`ConnectionError`** 进行自动重试。
- 发生可重试异常且次数未耗尽时，返回 `status="retrying"`，`step_retry_count += 1`；`step_index` 保持不变，`results` 不追加。
- 当前步骤成功后：`status="running"`、`error=None`、`step_retry_count=0`，然后按 `step_index` 继续或进入 Reviewer。
- 重试耗尽：`status="failed"`、保留简要错误信息并终止。
- `max_step_retries < 0`：构造时抛 `ValueError`。

### 4.3 Failure Handling 矩阵

| 场景 | 当前处理 |
|---|---|
| Planner 抛出异常 | 捕获并写入 `status="failed"`；不进入 Executor |
| Planner 返回空 Plan | 标记失败；不进入 Executor |
| Executor 抛出可重试网络/超时异常 | 在限制内重试同一 Step；耗尽后失败 |
| Executor 抛出其他 `Exception` | 直接失败，不重试 |
| Executor **明确返回** `status="failed"` | 直接失败，不因配置了 Retry 而重试 |
| Executor 返回无效的成功更新（无结果/索引未增加） | 标记失败 |
| Reviewer 返回 `approved=False` | 进入 Reject，错误原因使用 Reviewer Feedback |
| Reviewer 抛出异常或未返回有效 `ReviewResult` | 直接失败，不能继续通过 Review Router 抛异常 |
| 全部步骤完成且 Reviewer `approved=True` | Finish，设置 `completed` |

日志用于诊断异常，向 Workflow 调用者暴露的 `error` 应保持简短，不输出 API Key、请求正文等敏感信息。

### 4.4 当前实现的边界

1. **没有 Backoff/Jitter**：当前重试会立即重新进入 Executor，没有指数退避。
2. **没有持久化恢复**：进程中断、系统重启后不会自动从最近成功步骤恢复；Checkpointing 尚未纳入此阶段。
3. **没有 Replanning**：Reviewer 拒绝后直接结束，不根据反馈自动修改 Plan。
4. **没有端到端幂等保证**：即使异常类型是网络错误，也可能发生在外部工具成功执行之后；因此默认禁用 Step Retry，并应只在确认安全的场景中显式开启。
5. **没有语义级工具失败强判定**：如果 Agent 将某个工具错误转换为正常的非空回答，现有 Executor 可能将其视为成功；Reviewer 可以进行文本审核，但不能代替工具调用状态验证。
6. **没有覆盖所有异常类型**：其他框架包装异常、取消执行、Graph 配置/递归错误等未必转为 `WorkflowState.status="failed"`。
7. `recursion_limit` 当前按 Plan 最多五步和重试次数使用近似公式（例如 `max(30, 5 * (max_step_retries + 1) + 10)`）；这是 Graph 的执行安全阈值，**不是**业务 Retry 限制，也并非通用执行次数精确模型。

> 上述项目属于明确的实现限制或后续增强，并不意味着 Phase 6 必须一次性完成生产级可靠性工程。

## 5. LangGraph State、类型与运行入口的经验

- 讨论中明确采用 `dataclass WorkflowState`，以方便 Node 的属性访问。
- Node 实现使用 `async def __call__(state: WorkflowState) -> dict`，返回状态的局部更新。
- `results` 通过 reducer 追加；Graph 输出最终组装为 `WorkflowState` 供上层使用。
- 曾出现 Pylance 对 `self._graph.ainvoke(vars(state).copy(), ...)` 的静态类型错误：推断 Graph 需要 `WorkflowState`，实际传入 `dict[str, Any]`。最近讨论采用**传入 `state` dataclass 实例**的方式解决静态类型问题，并保留 `WorkflowState(**result)` 作为当前代码的输出转换方式。
- **注意**：类型检查通过不自动保证运行时兼容；在固定具体 LangGraph 版本下应使用 Graph Unit/Integration Tests 验证输入/输出形式。
- 开发环境方面，VS Code/Pylance 曾无法解析 `langchain_core`、`pydantic`，但 `uv run` 正常；通过 `Python: Select Interpreter` 重新选择项目 `.venv` 后解决。这属于 IDE 环境问题，不属于 Workflow 功能错误。

## 6. 测试设计与覆盖范围

### 6.1 测试目录分工

```text
tests/
├── unit/
│   ├── test_workflow_planner.py
│   ├── test_workflow_executor.py
│   ├── test_workflow_reviewer.py
│   └── test_workflow_graph.py
└── integration/
    └── test_workflow_graph.py
```

- `test_workflow_planner.py`：Mock 结构化 LLM 输出，检查 Plan 和 State 更新；**不连接 DeepSeek**。
- `test_workflow_executor.py`：Mock Runnable/Agent，检查步骤执行、前序结果上下文、索引更新和无效输出。
- `test_workflow_reviewer.py`：Mock 模型结构化响应，检查通过/拒绝以及执行完整性验证。
- `tests/unit/test_workflow_graph.py`：使用 FakePlanner、FakeExecutor、FakeReviewer 运行真实 LangGraph，验证编排控制与状态变化。
- `tests/integration/test_workflow_graph.py`：只保留真实 DeepSeek + LangChain Agent + Tools 的全链路测试。

### 6.2 Graph Unit Tests（设计为 11 个用例）

| 测试 | 主要断言 |
|---|---|
| `test_task_workflow_approved` | 两步按序执行、结果追加、Reviewer 调用一次、`completed`、输入 State 不被意外修改 |
| `test_task_workflow_executor_failure` | 显式失败立即结束；即使 `max_step_retries=2` 也不重试；Reviewer 不运行 |
| `test_task_workflow_review_rejected` | 步骤全部执行但 Reviewer 拒绝；`failed` 且保留 Feedback |
| `test_task_workflow_retry_success` | 第 0 步失败一次后成功，执行索引顺序为 `[0, 0, 1]`；不重复提交结果 |
| `test_task_workflow_retry_exhausted` | `max_step_retries=2`，实际尝试 `[0, 0, 0]`；状态失败 |
| `test_task_workflow_retry_disabled_by_default` | 默认首次失败即结束，不自动重试 |
| `test_task_workflow_planner_exception` | Planner 抛错后不调用 Executor/Reviewer |
| `test_task_workflow_empty_plan` | 空 Plan 立即失败 |
| `test_task_workflow_reviewer_exception` | 执行结果保留，Reviewer 自身异常标记失败 |
| `test_task_workflow_executor_unexpected_exception` | `ValueError` 等非重试异常只执行一次 |
| `test_task_workflow_negative_retries` | 负数重试次数构造失败 |

Fake Executor 在步骤完成时返回 `StepResult`；通过 `received_indices` 与 `attempted_indices` 两类记录分别区分**成功执行**和**总尝试次数**。

### 6.3 Graph Integration Test

`test_task_workflow_with_deepseek` 的主链路：

1. `load_llm_config()`（当前集成测试要求 provider 为 `deepseek`、`enable_thinking=False`）。
2. `LLMClient(config)` 初始化模型。
3. `load_tools()` 加载工具。
4. `PlannerNode(model=client.model, tools=tools)`。
5. `AgentFactory(client.model).build(tools=..., system_prompt=DEFAULT_SYSTEM_PROMPT, middleware=build_agent_middleware())`。
6. `ExecutorNode(agent=agent)` 与 `ReviewerNode(model=client.model)`。
7. `TaskWorkflow(...).run(WorkflowState(goal="Calculate 125 + 375 and explain the result."))`。
8. 检查 Plan、Results、Review、`completed`，以及执行结果包含 `500`。

真实 LLM 的步骤拆分和 Reviewer 输出存在不确定性；若 Integration Test 失败，需查看具体 Plan、Results、Feedback，不可直接认定 Graph 路由错误。

建议命令：

```bash
# 所有 Unit Tests；无需真实 DeepSeek API
uv run pytest tests/unit -v

# 仅运行 Graph Unit Tests
uv run pytest tests/unit/test_workflow_graph.py -v

# 显式启用真实 DeepSeek Graph Integration Test
RUN_DEEPSEEK_TESTS=1 uv run pytest tests/integration/test_workflow_graph.py -v -s
```

**验证状态**：已编写并讨论上述测试结构；本文整理时没有直接读取当前仓库或执行命令。尤其最近 `ainvoke(state, ...)` 的调整应再次运行测试确认。

## 7. 已确定但暂缓：Predefined Plan 输入

最初需求：某些任务的具体步骤由人工明确指定，不希望每次都让 LLM Planner 重新规划。

约定的最小设计：

```python
# Dynamic Plan（当前）
plan = await planner(state)

# Predefined Plan（未来入口，示例）
plan = [
    "Extract text from the image",
    "Search related documents",
    "Summarize the results",
]
```

**共同后续**：相同的 Executor → Reviewer → Finish/Reject 机制，不重写执行循环。

未来在进入 Planner 前增加分支：

```text
                     +--> PlannerNode --------+
User Request --------|                        +--> Executor -> Reviewer -> END
                     +--> Predefined Plan ----+
```

具体行为：

- `state.plan` 未提供：执行动态 Planner。
- `state.plan` 已提供且通过必要验证：跳过 Planner。
- 不能再无条件使用 `START -> planner`，否则会覆盖预定义 Plan。
- Predefined Plan 目前依然是**固定步骤 + Agent 动态选择工具**，并非完整的严格 Deterministic Workflow。

**当前状态：未实现；按后续讨论的决定，先完成 Retry / Failure Handling，再考虑此功能。** 不应因已经定义接口方向，就把它标为已完成。

## 8. 已确定但暂缓：User-defined Workflow Registration

### 8.1 最终采用的产品规则

讨论中曾提出一条路线：Agent 完成动态任务后，展示执行过的流程，询问用户是否保存，并自动泛化、注册。**此路线最终明确放弃**。

最终决定：

1. **仅注册用户明确提供并要求注册的 Workflow**；用户对持久化操作保有控制权。
2. 允许用户用 **TXT、Markdown、YAML、JSON** 等形式描述流程。
3. 对自然语言/半结构化描述，LLM 负责提取并转换为标准 Workflow Definition；**不应擅自修改用户明确指定的参数、步骤和约束**。
4. 对已经符合标准 Schema 的配置，优先直接解析与验证；不必无意义地调用 LLM。
5. 经 Schema、工具能力、参数/引用关系和权限验证后，**展示规范化定义，供用户确认或编辑**。
6. 用户确认后保存到 **Workflow Registry**，之后可以加载复用。
7. **LLM 自主生成的 Dynamic Plan 不自动保存，不提供将这次动态执行结果注册为模板的流程**。
8. 这套注册流程是普通任务执行之外的独立配置管理能力；本轮 Phase 6 不实现。

### 8.2 注册流程

```text
User-provided TXT / Markdown / YAML / JSON
                |
                v
   Workflow Parser / LLM Normalizer
                |
                v
    Standard Workflow Definition
                |
                v
 Schema / Capability / Permission Validation
                |
                v
      User Review and Confirmation
                |
                v
          Workflow Registry
                |
                v
       Load and Run When Selected
```

这里“从用户文本中提取流程”和“从 Agent 执行记录中归纳并自动注册流程”是**不同的功能**。只保留前者。

### 8.3 Plan 与 Workflow Definition 的区别

| 对象 | 内容 | 生命周期 |
|---|---|---|
| **Execution Plan** | 针对一次 Goal 的有序步骤，目前 `list[str]` | 单次任务；通常不持久化 |
| **Workflow Definition** | 用户定义的可复用流程，可能包含输入、步骤、明确工具、参数、依赖、条件、失败行为 | 用户确认注册后持久保存 |
| **WorkflowState** | 某一次执行实例的实时状态、结果、重试计数和 Review | 执行期间变化；当前尚无 Checkpoint 恢复 |

注册的对象应称为 **Workflow Definition**，而不是多个 Planner。**Planner 是生成执行计划的组件，Registry 管理的是用户定义的流程数据**。

### 8.4 可选组件边界（未来）

| 名称（暂定） | 责任 |
|---|---|
| `WorkflowParser` / `WorkflowNormalizer` | 解析用户输入；需要时让 LLM 转成标准格式 |
| `WorkflowDefinition` | 标准 Schema，包括输入、步骤、执行模式、工具绑定及约束 |
| `WorkflowValidator` | 验证结构、输入、工具存在性、权限、变量引用与依赖 |
| `WorkflowRegistry` | 保存、读取、列出、管理用户确认的 Workflow Definitions |
| `TaskWorkflow` / 执行引擎 | 执行已验证的定义，管理 Runtime State |

是否需要自动的 `PlanMatcher` 在最终设计中**不是必选项**。优先支持用户显式选择某个已注册 Workflow，避免让 LLM 无依据地决定所有任务应匹配哪个模板。

### 8.5 未来 Schema 草案（**仅为示意，尚未实现**）

```yaml
name: device_analysis
version: 1

inputs:
  image_id:
    type: string
    required: true

steps:
  - id: extract_model
    execution_mode: tool
    tool: visual_ocr
    arguments:
      image_id: "${inputs.image_id}"
    save_as: device_model

  - id: search_documents
    execution_mode: tool
    tool: rag_search_documents
    arguments:
      query: "${steps.extract_model.output}"
    save_as: documents

  - id: summarize
    execution_mode: agent
    instruction: "Summarize specifications using retrieved documents."
```

上述工具名、字段名、变量引用语法均为**示例占位**，并未验证与实际 MCP Tool Schema 的对应关系，也**不是**已冻结的正式接口。

严格执行用户指定工具与参数时，现有接受字符串 Prompt 的 `ExecutorNode` **不够**；未来需支持特定 Tool 的直接调用或独立执行 Node，并实现依赖、参数绑定、分支控制和错误规则。不能仅凭 YAML 文本保证确定性。

### 8.6 注册的安全与权限原则

- 用户上传的 Workflow Definition 属于**不可信输入**，不能直接执行任意 Python 代码或系统命令。
- LLM 只负责转换和辅助校验，不自行授予工具权限。
- 必须验证允许使用的 Tool、参数类型、变量引用以及当前环境是否具备相关能力。
- 涉及写入、删除、通知等副作用时要有明确授权，必要时保留人工确认（Human-in-the-Loop）。
- 已注册流程依赖的 MCP 服务不可用时，需明确报告能力不足，而不是编造执行结果。
- 新增业务流程一般新增配置，不应迫使通用 Agent 核心不断增加硬编码；但新增底层能力仍需要相应服务/工具支持。

## 9. 与 CVHub 其他子系统的架构边界

```text
CVHub Agent
├── Direct LangChain Agent
├── Task-level LangGraph Workflow
│   ├── Planner
│   ├── Executor (uses LangChain Agent)
│   ├── Reviewer
│   ├── Conditional Routing
│   └── Retry / Failure Handling
├── Tool Registration / MCP Clients
│   ├── Visual MCP / OCR Service
│   ├── RAG MCP / Search Service
│   └── Other external tools
└── [Later] User-defined Workflow Registry

External Services
├── OCR: its own detection / recognition Pipeline
└── RAG: its own retrieval / reranking Pipeline
```

明确的边界决定：

- **不把 OCR 的图片预处理、检测、识别等内部步骤再复制到 LangGraph**。OCR Service 本身负责其专用 Pipeline。
- **不把 BM25、Vector Search、RRF、Reranker 等 RAG 内部检索步骤写死在通用 Agent Graph 中**。RAG Service 保持独立。
- Agent 的 Workflow 负责决定**何时调用** OCR、RAG，如何组织跨服务任务和处理最终反馈。
- 后续如确有需要，可以另外实现明确的跨服务固定流程，但它应以业务配置/专用 Workflow 的方式存在，而不是扩张通用 TaskWorkflow 核心。
- Phase 5 的真实 OCR gRPC / RAG HTTP 工具接入可以独立推进；Phase 6 开发时允许先通过 Mock MCP 工具验证编排。

## 10. 讨论过但本阶段不做的功能

| 功能 | 决策 / 原因 |
|---|---|
| Agent 动态计划执行后自动建议保存为模板 | **已放弃**；易积累重复/错误模板，且缺乏用户明确的流程定义 |
| 从动态执行历史自动泛化为可复用模板 | **已放弃作为默认注册路线**；数据/参数固化风险高 |
| User-defined Workflow Parser / Registry | **保留设计、后续独立开发** |
| Predefined Plan 跳过 Planner | **已确定需求、暂缓实现** |
| 严格 Deterministic Tool-bound Workflow | **未来支持**；不同于目前的固定文字 Plan + Agent Executor |
| Reviewer 反馈驱动 Replanning | **后续增强**；需要管理旧结果、计划替换、次数及终止规则 |
| Step Retry Backoff / Jitter | **后续增强** |
| Workflow Checkpointer / Crash Recovery | **后续增强** |
| 副作用工具幂等与事务补偿 | **生产级增强** |
| 步骤级执行证据、结构化结果与评估指标 | **后续增强**，尤其适合 Phase 7 之后的可观测性与 Evaluation |

## 11. Phase 6 Aufgabenliste 与阶段成果

此处按讨论时的实际范围划分。测试文件已经讨论并提供，但是否全部通过仍以仓库实际运行结果为准。

- [x] 学习 LangGraph StateGraph、Nodes、Edges、State 和 Reducer。
- [x] 设计并实现通用 `TaskWorkflow`（Planner → Executor → Reviewer）。
- [x] 在 Executor Node 中复用已有 LangChain Agent，支持动态 Tool Calling。
- [x] 实现 Conditional Routing、步骤循环、审核通过/拒绝和失败终止分支。
- [x] 实现基础 Hybrid Workflow（固定编排 + LLM 动态决策）。
- [x] 编写 Node Unit Tests、Graph Unit Tests、DeepSeek Integration Test。
- [x] 实现基础 Workflow Step Retry 与 Failure Handling（**以最新版本运行测试为最终验收条件**）。
- [ ] 支持 Dynamic Plan / Predefined Plan 两种输入方式——**暂缓到后续迭代**。
- [ ] 运行最新所有 Unit Tests 与 DeepSeek Integration Test，确认最近的 Graph/State 修改没有引入回归——**如尚未执行，仍需完成**。

**当前实现的阶段成果**：

1. **Direct Agent**：已有 LangChain Agent，可以直接处理用户请求。
2. **Dynamic Plan Workflow**：通过 Planner → Executor → Reviewer 完成任务级编排。
3. **Hybrid Execution**：LangGraph 负责控制逻辑，Agent 负责步骤内部自主执行与工具选择。

**未来的第三种用户入口（目前未实现）**：**Predefined Plan Workflow**。其步骤由调用方或已注册定义提供，跳过 Planner，仍可复用 Executor/Reviewer。

## 12. 后续文档与开发时的建议

写正式架构文档时可按以下顺序展开：

1. **Why**：为什么 CVHub 需要任务级 Workflow，而不是直接让 Agent 自由执行所有任务。
2. **Architecture**：StateGraph、State、Nodes、Edges、Reducer、Conditional Routing。
3. **Execution**：Planner、Executor、Reviewer 的输入输出与流程。
4. **Reliability**：Retry 分类、失败传播、幂等风险、未覆盖边界。
5. **Validation**：Unit 与 Integration 分层、测试用例及验证命令。
6. **Extensibility**：Dynamic vs. Predefined Plan、未来 Workflow Definition / Registry。
7. **Boundaries**：Agent 编排与 OCR/RAG 服务内部 Pipeline 的职责区别。

开发后续功能时建议保持以下约束：

- **先统一 Execution Plan 的输入边界**，再扩展更多 Plan 来源；不要复制第二套 Executor/Reviewer。
- 新流程的**注册**与一次任务的**执行**是两个操作，使用不同接口与权限控制。
- 不要把未经确认的 LLM 动态 Plan 自动注册为持久化资产。
- 当前 `list[str]` 足以支持已有 Dynamic Plan；只有在明确需要工具绑定、参数与条件分支时才演进到结构化 Step Schema。
- 任何 Workflow 成功状态必须结合 Review 与可观察执行结果理解，不把 LLM 的肯定反馈当作外部事实证明。

---

## 归档结论

Phase 6 的核心决策是：**用 LangGraph 管理通用任务编排与失败控制，用现有 LangChain Agent 执行各步骤，用 MCP 连接独立 OCR/RAG 等服务。**

在此基础上，当前优先交付 **Dynamic Plan Workflow + Hybrid Execution + 基础 Retry/Failure Handling**；**Predefined Plan** 作为下一步接口扩展；**User-defined Workflow Registration** 作为未来独立模块，仅保存用户明确提交并确认的流程定义，不自动保存 LLM 自主规划出的临时方案。
