# CvHub Agent 技术选型讨论总结

## 1. 当前 Agent 开发范围

当前 CvHub 的 RAG System 计划作为独立仓库 / 独立容器开发。

因此当前 Agent 项目的重点是 AI Agent System 本身，包括：

- Agent Orchestrator
  - Intent Analysis
  - Task Planning
  - Execution Scheduling
  - Result Composition
- Tool / Skill Management
  - Skill Registry
  - Capability Index
  - Service Discovery
  - Metadata & Schema Store
- Memory
- Safety / Permission / Audit
- 与 LLM System、RAG System、外部微服务之间的接口

视觉能力如 OCR、Object Detection、Feature Matching、QR/Barcode 等继续保持为独立微服务，Agent 不直接包含这些算法实现。

---

## 2. 编程语言

### 推荐：Python 3.12

原因：

- LLM / Agent / RAG 生态成熟
- LangGraph、Pydantic、PydanticAI 等主要框架均以 Python 为核心
- 与现有 CvHub OCR、LLM、RAG 方向保持一致
- gRPC、FastAPI、asyncio 等基础设施成熟
- 适合快速迭代，同时也可以保持良好的工程结构

---

## 3. OpenCode

OpenCode 不是典型意义上的 Agent 开发框架。

更准确地说，它是一个已经实现好的 Coding Agent / Agent Runtime。

典型使用方式：

```text
LLM
 ↓
OpenCode
 ↓
Agent Loop / Tool Calling / Permission / Session
 ↓
MCP
 ↓
External Tools
```

如果使用 OpenCode，通常只需要：

1. 配置自己的 LLM
2. 配置 Agent instructions
3. 提供 MCP tools
4. 让 OpenCode 完成 Agent loop、tool selection、session 等工作

### 对 CvHub 的判断

OpenCode 可以快速构建一个可工作的 Agent，但它已经替开发者实现了大量 Agent Core。

CvHub 当前目标是自己设计和实现：

- Orchestrator
- Planning
- Tool Management
- Memory
- Safety
- Execution Flow

因此 OpenCode 不适合作为 CvHub Agent 的核心框架。

不过它的 Tool、Permission、Session、Agent 配置等设计可以作为架构参考。

---

## 4. LangGraph

LangGraph 是专门面向 Agent Workflow / Orchestration 的框架。

核心概念：

```text
State
Node
Edge
Conditional Edge
Graph
```

Agent 可以表示为：

```text
START
  ↓
Intent Analysis
  ↓
Task Planning
  ↓
Tool Execution
  ↓
Result Composition
  ↓
END
```

也可以构建复杂流程：

```text
Planner
   ↓
Execute Tool
   ↓
Check Result
   ├── Success → Composer
   └── Failure → Replan → Execute Tool
```

或者：

```text
              ┌→ OCR ────────┐
Planner ──────┼→ Detection ──┼→ Merge
              └→ RAG ────────┘
```

### 优点

- 非常适合复杂 Agent workflow
- 原生支持状态管理
- 支持条件分支
- 支持循环和重新规划
- 适合多步骤 Tool Execution
- Agent 流程非常清晰
- 和 CvHub 当前设计的 Orchestrator 高度匹配

### 缺点

- 学习成本比普通 Python Agent 高
- 会引入 Graph / Node / State 等框架概念
- 如果 Agent 很简单，可能显得偏重

### 对 CvHub 的适配

CvHub 当前已经明确设计了：

```text
Intent Analysis
Task Planning
Execution Scheduling
Result Composition
```

这本身就非常接近一个 Agent Graph。

因此 LangGraph 是目前最符合 CvHub Agent 架构的候选框架。

---

## 5. PydanticAI

PydanticAI 是一个强调 Python 类型系统、结构化输出和 Tool Calling 的 Agent Framework。

主要思想：

```text
Agent
 ↓
LLM
 ↓
Tool Calling
 ↓
Pydantic Validation
 ↓
Structured Result
```

例如：

```python
class PlannedTask(BaseModel):
    tool_name: str
    arguments: dict
    depends_on: list[str]
```

这样可以要求 LLM 返回确定的数据结构，而不是自然语言描述。

### 优点

- Pythonic
- 类型安全
- Structured Output 非常自然
- Tool Schema 定义简单
- 强大的 Pydantic Validation
- 开发普通 Tool Agent 很方便
- 框架侵入性相对较小

### 缺点

- 复杂 Agent workflow 不是它最核心的优势
- 如果需要复杂分支、循环、重试、重新规划，LangGraph 更自然
- CvHub 当前的 Orchestrator 设计偏 workflow-oriented

---

## 6. LangGraph 与 PydanticAI 的核心区别

可以简单理解为：

### LangGraph

> Workflow first

关注：

- Agent 怎么走
- 当前 State 是什么
- 下一步执行哪个 Node
- 是否需要分支
- 是否需要重试
- 是否需要重新 Planning

更接近：

```text
Agent Execution Engine
```

---

### PydanticAI

> Agent + Types first

关注：

- LLM 如何调用 Tool
- Tool 参数是什么
- 输出结构是什么
- 如何验证 LLM 输出
- 如何让 Agent 以 Python 类型安全的方式运行

更接近：

```text
Typed Agent Framework
```

---

## 7. Pydantic 与 PydanticAI 要区分

即使不使用 PydanticAI，CvHub Agent 仍然非常适合使用 Pydantic。

例如：

```text
IntentResult
TaskPlan
PlannedTask
ToolDefinition
ToolCall
ToolResult
AgentResponse
AgentConfig
```

这些数据结构都可以使用普通 Pydantic BaseModel。

因此：

```text
Pydantic
```

和：

```text
PydanticAI
```

不是同一个技术选择。

---

## 8. 当前推荐技术栈

目前最推荐：

```text
Python 3.12
│
├── FastAPI
│     └── REST API
│
├── LangGraph
│     └── Agent Orchestration
│
├── Pydantic
│     ├── State Models
│     ├── Task Models
│     ├── Tool Schema
│     └── Agent Response
│
├── grpc.aio
│     └── Visual Microservices
│
├── httpx
│     └── LLM / RAG Service
│
├── asyncio
│     └── Async Execution
│
├── pytest
│     └── Testing
│
└── Docker
      └── Deployment
```

第一版暂时不引入 PydanticAI。

---

## 9. 推荐的职责边界

### FastAPI

负责：

```text
Client
 ↓
REST API
 ↓
Agent
```

只处理 API 层。

---

### LangGraph

负责：

```text
Intent
 ↓
Planning
 ↓
Execution
 ↓
Routing
 ↓
Retry / Replan
 ↓
Composition
```

只负责 Agent orchestration。

---

### Pydantic

负责：

```text
Data Model
Validation
Schema
Structured Output
```

---

### Tool Management

建议仍然由 CvHub 自己开发：

```text
ToolRegistry
ToolDefinition
CapabilityIndex
ToolExecutor
ServiceDiscovery
```

LangGraph 只调用 ToolManager，不负责管理整个 Tool 系统。

例如：

```text
LangGraph
    ↓
ToolExecutor
    ↓
ToolRegistry
    ↓
gRPC Client
    ↓
OCR Service
```

---

## 10. 当前技术选型优先级

### 第一选择

```text
LangGraph + Pydantic
```

原因：

CvHub Agent 本身已经明确设计为：

```text
Intent Analysis
Task Planning
Execution Scheduling
Result Composition
```

后续很可能出现：

- Tool dependency
- 条件执行
- 多 Tool workflow
- Parallel Tool Execution
- Retry
- Replan
- Result Validation

LangGraph 对这些场景非常适合。

---

### 第二选择

```text
Pure Python + Pydantic
```

优势是完全掌控代码和架构。

如果第一版 Agent 非常简单，也可以先自己实现 Agent Loop。

但是随着 workflow 复杂度增加，需要自己实现：

- State management
- Routing
- Retry
- Workflow
- Resume
- Branching

长期维护成本可能上升。

---

### 第三选择

```text
PydanticAI
```

适合：

- Tool Calling
- Structured Output
- 相对简单的 Agent
- 强类型 Python Agent

但当前 CvHub Agent 更强调 Orchestration，因此优先级低于 LangGraph。

---

## 11. 当前结论

当前推荐方向：

```text
Language:
Python 3.12

API:
FastAPI

Agent Orchestration:
LangGraph

Data Models:
Pydantic

Microservice Communication:
grpc.aio

LLM / RAG HTTP Communication:
httpx

Async:
asyncio

Testing:
pytest

Deployment:
Docker
```

核心原则：

> LangGraph 管 Agent 怎么执行。

> Pydantic 管数据长什么样。

> Tool Management 由 CvHub 自己实现。

> OCR / Detection / RAG / LLM 等能力继续作为独立服务存在。

这样既能使用成熟 Agent Framework，又不会让 CvHub 的核心架构完全被框架接管。
