# CVHub Agent — Task List

**当前目标：** 尽快打通 Agent 平台的最小可用链路，同时保留 MCP、LangGraph、RAG、Evaluation 与工程化任务。

**近期执行顺序：** 先完成 Phase 3 的 Agent Factory 与 Native Tools 基础结构（暂不修改模型适配）；之后完成 Phase 2 的 DeepSeek 最小调用验证 → Phase 3 Agent 运行测试 → Phase 8 基础 API（提前）→ Phase 4 MCP → Phase 5 外部服务 → Phase 6 Workflow。其余任务在后续阶段补齐。

> 模型策略：当前优先 DeepSeek；本地 vLLM / Qwen 暂缓但不删除。后续统一支持两者，不在本轮处理具体的模型适配实现。

## Phase 1 — Project Setup & Architecture
- [x] 创建 cvhub-agent Repository，使用 uv 管理 Python 环境和依赖
- [x] 确定 LangChain、LangGraph、FastAPI、MCP SDK 等技术栈及版本
- [x] 设计项目结构及 Agent、Workflow、Tools、MCP、Clients 的边界
- [x] 实现 Configuration Management（.env、Pydantic Settings、Secrets）
- [x] 配置 Ruff、pytest、Git 和基础 CI

阶段成果：可维护、可测试的 Agent 项目基础结构

## Phase 2 — LLM Serving & Integration

### 当前优先 — DeepSeek API
- [x] 注册 DeepSeek 账户，并在 `.env` 中配置 API Key
- [x] 复用统一 LLM Client，接入 DeepSeek OpenAI-compatible API 并验证基本 Chat
- [x] 配置 Model、Temperature、Max Tokens、Timeout、Retry 等参数
- [x] 验证 Tool Calling、Structured Output 和 Streaming
- [x] 测试 API 超时、调用失败、限流及错误反馈
- [x] 记录 Token Usage 与 API 调用成本

### 暂缓 — 本地 vLLM / Qwen（保留原任务）
- [x] ~~配置并启动现有 vLLM + Qwen 推理服务~~
- [ ] ~~实现统一的 LLM Client，支持 OpenAI-compatible API~~
- [ ] ~~实现 Model、Temperature、Context Length、Timeout 等配置~~
- [ ] ~~验证 Chat、Tool Calling、Structured Output 的兼容性~~
- [ ] ~~设计基础推理 Benchmark，记录 Latency、Throughput、显存占用~~
- [ ] ~~测试模型服务不可用及请求超时的处理~~

~~原阶段成果：经过兼容性验证的独立 LLM 推理服务~~

**后续设计要求：** Agent 层应独立于具体模型提供方；后续支持 DeepSeek API 与本地 vLLM 两种运行模式。当前不实现双后端切换，不展开模型参数兼容性改造。

阶段成果：DeepSeek API 能提供基本对话和 Agent Tool Calling；本地模型接入留待后续。

## Phase 3 — LangChain Core Agent
- [x] 使用 LangChain create_agent() 创建 Agent Factory（模型实例由外部注入，不绑定 DeepSeek 或 vLLM）
- [x] 定义 System Prompt、Agent 配置及调用入口
- [x] 使用 @tool 实现 LangChain Native Tools
- [x] 测试 Tool Schema、Tool Calling、工具结果反馈
- [x] 测试 Multi-step Tool Calling、Structured Output，并设置 Model / Tool Call 次数上限及停止条件
- [x] 编写 Agent Unit Tests，Mock 外部模型与工具
- [x] 完成最小运行验证：用户请求 → Agent → Native Tool → 最终回答

阶段成果：能够调用 Native Tools 的基础 Agent

## Phase 4 — MCP Integration
- [x] 学习 MCP 基础架构、Tools、Resources、Prompts 和 Transport 的职责
- [x] 使用 MCP Python SDK 实现独立 MCP Server
- [x] 通过 Streamable HTTP 暴露两个实际可调用的 MCP Tools
- [x] 集成 LangChain MCP Client Adapter，完成 Tool Discovery 与调用
- [x] 实现 MCP Tools 与 Native Tools 的统一加载
- [x] 测试 MCP 参数验证、网络断连、Timeout 和错误处理
- [x] 验证 MCP Server 独立进程及 Docker 网络通信

阶段成果：完整的 MCP Server → MCP Client → Agent 工具调用链路

## Phase 5 — External Services & RAG
- [ ] 开发 RAG Client，调用 CVHub RAG 检索接口
- [ ] 通过 MCP Server 暴露 RAG Search Tool
- [ ] 开发 OCR gRPC Client，并封装 OCR Tool
- [ ] 设计工具输入输出 Schema，包括文档标识、结果及引用来源
- [ ] 验证 Agent 能选择工具、组合调用并处理真实结果
- [ ] 处理外部服务失败、无检索结果、数据过长等情况

阶段成果：Agent 与已有 RAG、OCR 微服务协同工作

## Phase 6 — Workflow & Orchestration

- [x] 学习 LangGraph StateGraph、Nodes、Edges、State 和 Reducer
- [x] 实现通用 TaskWorkflow（Planner → Executor → Reviewer）
- [x] 在 Executor Node 中复用现有 LangChain Agent，支持动态 Tool Calling
- [x] 实现 Conditional Routing、步骤循环、执行完成与审核失败分支
- [x] 实现基础 Hybrid Workflow（固定编排流程 + LLM 动态规划与决策）
- [x] 编写 Unit Tests 和 Integration Tests，验证执行顺序、State 更新及基本错误路径
- [ ] 支持 Dynamic Plan 和 Predefined Plan 两种输入方式，统一使用现有 Executor 和 Reviewer
- [x] 完善 Workflow 级别的 Retry 和 Failure Handling，包括重试次数、异常处理与失败终止
- [x] 补充 Predefined Plan、Retry 和 Failure Handling 的相关测试

阶段成果： 支持 Direct Agent、Dynamic Plan Workflow、Predefined Plan Workflow 三种任务执行模式，并通过 Hybrid Workflow 结合确定性流程控制与 Agent 自主决策。

## Phase 7 — Memory & Context Engineering
- [ ] 通过 LangGraph Checkpointer 实现 Short-term Memory（可在 Phase 3 后先用 InMemorySaver 验证多轮会话）
- [ ] 使用 thread_id 实现会话隔离和恢复
- [ ] 使用 PostgreSQL Checkpointer 实现跨进程持久化
- [ ] 实现 Conversation History 管理及 Context Trimming / Summarization
- [ ] 管理 System Prompt、RAG Context、Tool Results 的输入内容
- [ ] 测试长对话、Context 超限和不同会话的状态隔离

阶段成果：具备多轮对话、状态恢复和上下文控制的 Agent

## Phase 8 — Backend API & Security

**执行顺序：** 以下基础 API 任务提前至 Phase 3 完成后，其余安全与治理任务仍在本阶段完善。

- [ ] 【提前】使用 FastAPI 实现 Chat API、Streaming API 和 Health Check
- [ ] 【提前】通过 Pydantic 定义并验证 Request / Response Schema
- [ ] 实现必要的 API Authentication、配置及密钥管理
- [ ] 为 Tools 添加 Allowlist、参数限制和操作权限
- [ ] 处理 Prompt Injection 风险，区分不可信内容与系统指令
- [ ] 添加 Timeout、Rate Limiting、Error Handling 和 Request ID

阶段成果：能够被客户端安全访问的 Agent Backend

## Phase 9 — Evaluation & Observability
- [ ] 建立 Agent Evaluation Dataset，包含正常及异常任务
- [ ] 评估 Tool Selection、Tool Arguments、Task Success Rate
- [ ] 对 Workflow 和 RAG 结果进行端到端回归测试
- [ ] 集成 Langfuse 或 LangSmith，记录 Agent Execution Traces
- [ ] 记录 Latency、Token Usage、LLM / Tool 错误及必要的成本指标
- [ ] 建立 Unit、Integration、E2E Tests，并接入 GitHub Actions CI

阶段成果：具有可重复评估、追踪及回归测试能力的系统

## Phase 10 — Docker & Cloud Deployment
- [ ] 为 Agent Backend 和 MCP Server 创建 Dockerfile
- [ ] 编写 Docker Compose，配置容器网络、环境变量和 Health Check
- [ ] 集成 DeepSeek API、RAG、OCR 服务的连接配置，并保留后续本地 vLLM 接入能力
- [ ] 完成容器环境中的 End-to-End Tests
- [ ] （后续）选择 Azure 或 AWS，实践一次 Agent API + MCP 服务部署
- [ ] 编写 Architecture、Deployment、Testing 和 API Documentation

阶段成果：能够独立部署、可重复测试的 CVHub Agent 系统


