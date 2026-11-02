# CVHub Agent — Memory & Context Engineering 总结

> **项目**：CVHub Agent  
> **阶段**：Phase 7 — Memory & Context Engineering  
> **整理日期**：2026-11-02  
> **定位**：技术设计及开发归档。区分已实现功能与后续计划，供正式技术文档编写时参考。

## 1. 设计目标

Memory 系统需要解决三个核心问题：

| 目标 | 说明 |
|---|---|
| **Short-term Memory** | Agent 记住同一会话中的历史消息，实现多轮对话 |
| **State Persistence** | Agent 和 Workflow 的状态可以持久化，在重新连接或重启后恢复 |
| **Context Management** | 控制每次发送给 LLM 的上下文，降低 Token 消耗，避免 Context Window 超限 |

**当前进度**：Short-term Memory 和基础 State Persistence 已完成并经过相应测试；Context Management 尚未开始实现。

## 2. 核心概念

| 概念 | 职责 |
|---|---|
| **Memory** | Agent 保存、复用历史信息的能力 |
| **State** | 当前状态数据，如 Messages、Plan、Results |
| **Checkpoint** | 某一执行边界保存的 State 快照 |
| **Checkpointer** | LangGraph 提供的状态保存与恢复接口 |
| **Thread** | 连续对话或一次 Graph 执行的状态历史 |
| **thread_id** | 定位、隔离并恢复特定 Thread 的标识 |
| **Persistence** | 将状态保存至 PostgreSQL 等持久化存储 |
| **Context** | 当前实际传给 LLM 的输入内容 |
| **Context Management** | 选择、裁剪、压缩与组织模型输入 |
| **Long-term Memory** | 跨会话保存长期事实、偏好及经验的能力（暂缓） |

**核心区别**：Memory 决定系统保存什么，Context Management 决定当前给 LLM 发送什么。数据库中存有完整对话历史，并不意味着每轮都需要把全部历史发给模型。

### 2.1 Checkpointer、Checkpoint 和数据库之间的关系

- `WorkflowState` / Agent State：记录当前任务或聊天的数据。
- `Checkpoint`：某个时刻的状态快照。
- `BaseCheckpointSaver`：Checkpointer 的统一接口。
- `InMemorySaver`：把快照保存在当前进程内存中。
- `AsyncPostgresSaver`：把快照持久化到 PostgreSQL。
- `thread_id`：选择属于哪次会话或 Workflow Run 的快照序列。

Checkpointer **不负责**历史信息摘要、Token 预算或语义检索。

## 3. 当前 Memory 架构

```text
Direct Agent                       TaskWorkflow
(Conversation Messages)           (Plan / Results / Status)
       |                                   |
       +---------------+-------------------+
                       |
              BaseCheckpointSaver
                       ^
                       |
          persistence/checkpointer.py
                 open_checkpointer()
                       |
                +------+-------+
                |              |
          InMemorySaver   AsyncPostgresSaver
             (RAM)          (PostgreSQL)
```

采用 **Dependency Injection（依赖注入）**：

- `AgentFactory`、`TaskWorkflow` 都可接收 `BaseCheckpointSaver`，不负责创建数据库连接。
- `open_checkpointer()` 统一选择 Backend、管理连接生命周期。
- Conversation 和 Workflow Run 使用不同的 `thread_id`，避免状态相互混淆。
- 有持久化需求的 Agent / Workflow 操作必须在 Checkpointer 连接仍然有效的生命周期内完成。

## 4. 已实现的模块

### 4.1 统一 Checkpointer 管理

**路径**：`src/cvhub_agent/persistence/checkpointer.py`

```python
CheckpointBackend = Literal["memory", "postgres"]

@asynccontextmanager
async def open_checkpointer(
    backend: CheckpointBackend,
    *,
    database_url: str | None = None,
) -> AsyncIterator[BaseCheckpointSaver]:
    ...
```

| Backend | 实现 | 用途 |
|---|---|---|
| `memory` | `InMemorySaver` | 本地调试、Unit Tests、临时会话 |
| `postgres` | `AsyncPostgresSaver` | 持久化会话、Workflow 状态及恢复 |

示例：

```python
async with open_checkpointer(
    "postgres",
    database_url=database_url,
) as checkpointer:
    # Build and use Agent / Workflow while connection is open.
    ...
```

数据库 Schema 初始化不属于这一运行时模块，由独立初始化脚本完成。

### 4.2 Direct Agent Short-term Memory

**路径**：`src/cvhub_agent/agent/factory.py`

在原有 `build()` 方法中添加可选参数：

```python
checkpointer: BaseCheckpointSaver | None = None
```

并传入 LangChain Agent 的创建函数。使用同一个 `thread_id` 发起后续请求，Agent 可以恢复该 Thread 先前保存的历史 Messages，不需要调用方人工拼接全部历史。

```python
thread_config: RunnableConfig = {
    "configurable": {"thread_id": "conversation-001"}
}
```

不同 `thread_id` 具有独立的消息状态；真正对外提供服务时仍需在应用层检查用户对 Thread 的访问权限。

### 4.3 PostgreSQL Persistence

CVHub Agent 独立部署了 PostgreSQL 数据库，与 CVHub RAG 的数据库、Volume 及 Compose 项目分离。

| 项目 | 当前设计 |
|---|---|
| PostgreSQL 版本 | 17 |
| Database | `cvhub_agent` |
| Docker Compose Project | `cvhub-agent` |
| 宿主机端口 | `18010` |
| 容器端口 | `5432` |
| 数据持久化 | 独立 Docker Volume |
| 初始化脚本 | `scripts/init_checkpoint_db.py` |

初始化调用：

```python
async with AsyncPostgresSaver.from_conn_string(
    database_url
) as checkpointer:
    await checkpointer.setup()
```

LangGraph 创建并维护以下表：

```text
checkpoint_migrations
checkpoints
checkpoint_blobs
checkpoint_writes
```

无需自己设计 Checkpointer 的数据库 Schema。`setup()` 用于初始化或迁移，而不是每轮对话请求都执行。

将来部署 Agent Docker 容器时，可在同一个 Compose 文件中加入 Agent 服务，并使用数据库健康检查依赖，实现联合启动。容器间使用 `db:5432`，本地开发环境从宿主机使用 `127.0.0.1:18010`。

### 4.4 TaskWorkflow Persistence

**路径**：`src/cvhub_agent/workflows/workflow.py`

`TaskWorkflow` 同样支持可选 Checkpointer：

```python
checkpointer: BaseCheckpointSaver | None = None
```

编译 Graph 时注入：

```python
builder.compile(checkpointer=self._checkpointer)
```

保存的 `WorkflowState` 主要包括：

```text
WorkflowState
├── goal
├── plan
├── step_index
├── results
├── status
├── error
├── review
└── step_retry_count
```

几个重要的 LangGraph API：

| API | 用途 |
|---|---|
| `graph.ainvoke()` | 执行或从可恢复位置继续 Graph |
| `graph.aget_state()` | 读取指定 Thread 的最新 `StateSnapshot` |
| `graph.aget_state_history()` | 读取指定 Thread 的历史 Checkpoints |

`StateSnapshot.values` 是保存的状态，`StateSnapshot.next` 是等待执行的 Node 列表。`next == ()` 仅说明没有待执行 Node，不代表业务一定成功；业务结果应另外检查 `status`。

### 4.5 Workflow Pause / Resume

已为 `TaskWorkflow` 添加暂停和恢复接口。

启动时可在 Executor 完成后暂停：

```python
await workflow.run(
    state,
    thread_id=thread_id,
    pause_after_executor=True,
)
```

重新建立连接并创建新的 `TaskWorkflow` 后，使用同一个 `thread_id` 继续：

```python
await workflow.resume(thread_id=thread_id)
```

底层恢复使用：

```python
await graph.ainvoke(None, config=config)
```

**不是**重新提交最初的 `WorkflowState`。测试所覆盖的流程如下：

```text
Planner       Completed
Executor 0    Completed
                 |
               Pause
                 |
       Close DB Connection
                 |
       Reconnect / Resume
                 |
Executor 1    Completed
Reviewer      Approved
Finish        Completed
```

测试证明重新连接后，已完成的 Planner 和 Executor Step 0 不被重新执行。当前使用 LangGraph 静态断点；以后如需 Human-in-the-loop 审批，可以扩展动态 `interrupt()` 机制。

## 5. 测试情况

| 测试 | 验证内容 | 状态 |
|---|---|---|
| InMemorySaver Integration | DeepSeek 多轮对话继承 | 已通过 |
| Thread Isolation Unit | 不同 Thread 消息隔离 | 已通过 |
| PostgreSQL Agent Integration | 重新连接、新建 Agent 后恢复消息 | 已通过 |
| Workflow Checkpoint Unit | Graph State 及 Checkpoint History | 已通过 |
| Workflow PostgreSQL Integration | 数据库持久化 | 已通过 |
| Workflow Resume Integration | 暂停、重连、恢复和执行顺序 | 已通过 |

主要测试文件：

```text
tests/
├── unit/
│   ├── test_agent_memory.py
│   └── test_workflow_graph.py
└── integration/
    ├── test_agent_memory.py
    └── test_workflow_persistence.py
```

PostgreSQL 测试使用唯一的测试 Thread ID，通过 `finally` 与 `adelete_thread()` 清理该测试创建的数据，避免影响其他会话或保留垃圾测试记录。

需要注意：目前测试验证了连接重建、Graph 实例重建及显式 Pause/Resume；这**不等于**已证明突发操作系统进程崩溃时的所有副作用都能安全恢复。

## 6. 重要设计决策与边界

1. **Checkpointer 不是 Memory Manager**：只负责状态持久化和恢复，不负责历史摘要、Token Budget 或语义记忆检索。
2. **Conversation State 与 Workflow State 隔离**：可共享 Saver 类型，但各自使用独立 `thread_id`；不能自动假定两者共用聊天上下文。
3. **Persistence 不等于 Context Compression**：数据库存储状态与 LLM 实际接收的输入可以不同。
4. **Resume 不等于外部工具幂等**：若工具已写入外部系统，但对应 Node 尚未成功提交 Checkpoint，恢复时可能重复调用工具。
5. **默认异步 Checkpoint 持久化**：测试中曾因显式使用 `durability="sync"` 遇到 LangGraph 内部 `TypeError: 'method' object can't be awaited`；移除显式参数后相关测试通过。突发崩溃时，异步持久化可能丢失最近尚未完成写入的快照。
6. **不自研 Checkpoint Repository**：使用 LangGraph 官方 `InMemorySaver` / `AsyncPostgresSaver`，通过轻量接口统一管理连接。
7. **不重复启动已经使用过的 Workflow Thread**：`results` 采用追加式 Reducer，旧状态与新的初始 State 混用可能造成重复结果。
8. **资源生命周期明确**：长期在线应用中，数据库连接的打开/关闭应由应用生命周期负责，而不是为每个工具调用临时创建并立即释放。

## 7. 尚未开发的 Context Engineering

| 功能 | 目标 |
|---|---|
| **Token Counting** | 估算或统计输入的 Token 数量 |
| **Context Budget** | 为 System Prompt、History、RAG、Tool Results、输出预留空间 |
| **Conversation Trimming** | 根据规则裁剪旧消息和不必要的数据 |
| **Conversation Summarization** | 将较长历史概括成摘要，并保留近期消息 |
| **Context Assembly** | 组织 System Prompt、Conversation、Workflow Results、RAG Context、Tool Results |
| **Long-context Tests** | 验证长对话、Context 超限、压缩后重要信息保留及 Token 改善情况 |

已确定的设计目标是：**在控制 Token 消耗的同时，尽可能保留当前任务需要的信息。**

推荐未来采用：

```text
System Prompt
+ Conversation Summary
+ Recent Messages
+ Current Question
+ Relevant Workflow / Tool / RAG Context
```

而不是每一轮都向模型发送完整历史。

**Long-term Memory**（跨 Thread 的长期事实与偏好、语义记忆检索）暂不作为本次已完成功能，也不在当前阶段优先实现。

## 8. Phase 7 当前进度

- [x] 明确 Agent Conversation State 与 TaskWorkflow State 的职责
- [x] 实现 `InMemorySaver` Short-term Memory
- [x] 实现 `thread_id` 多轮对话与会话隔离
- [x] 部署 Agent 独立 PostgreSQL，初始化 Checkpoint 表
- [x] 使用 `AsyncPostgresSaver` 实现基础持久化
- [x] 实现统一的 `open_checkpointer()` 接口
- [x] 实现 TaskWorkflow Checkpoint 保存
- [x] 实现 TaskWorkflow Pause / Resume
- [x] 验证消息恢复、Checkpoint History 与 Workflow 执行位置恢复
- [ ] 实现 Conversation Trimming / Summarization
- [ ] 实现 Context Budget / Token Management
- [ ] 统一管理不同来源的 Context
- [ ] 编写长上下文和压缩效果测试

### 当前阶段成果

CVHub Agent 已具备基础 **Short-term Memory、Thread Isolation、PostgreSQL State Persistence 与 TaskWorkflow Pause / Resume** 能力。剩余工作主要集中于 **Context Engineering**，尤其是 Token Budget、History Trimming、Conversation Summarization，以及不同 Context 来源的组织方式。
