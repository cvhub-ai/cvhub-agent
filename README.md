# Project Structure

```text
agent/
├── src/
│   ├── core/                     # Core Agent runtime
│   │   ├── runtime/              # Agent execution lifecycle
│   │   ├── planning/             # Task planning and decision making
│   │   ├── context/              # Context construction
│   │   ├── state/                # Runtime state management
│   │   ├── prompts/              # Prompt templates and management
│   │   └── guardrails/           # AI safety and execution constraints
│   │
│   ├── memory/                   # Agent memory system
│   │   ├── short_term/           # Conversation and runtime memory
│   │   ├── long_term/            # Persistent cross-session memory
│   │   ├── retrieval/            # Memory search and selection
│   │   └── storage/              # Memory persistence implementations
│   │
│   ├── rag/                      # Retrieval-Augmented Generation
│   │   ├── ingestion/            # Document loading and processing
│   │   ├── indexing/             # Index creation and updates
│   │   ├── retrieval/            # Search, filtering and reranking
│   │   ├── storage/              # Document and vector storage
│   │   └── citations/            # Source and citation management
│   │
│   ├── tools/                    # Agent tool system
│   │   ├── definitions/          # Tool definitions and schemas
│   │   ├── registry/             # Tool registration and discovery
│   │   ├── execution/            # Tool execution
│   │   └── validation/           # Tool input and output validation
│   │
│   ├── integrations/             # External service integrations
│   │   ├── llm/                  # LLM service client
│   │   ├── embeddings/           # Embedding service client
│   │   └── external_services/    # Other external systems
│   │
│   ├── api/                      # External API layer
│   │   ├── http/                 # REST API
│   │   ├── streaming/            # SSE and WebSocket communication
│   │   ├── middleware/           # API middleware
│   │   └── schemas/              # Request and response schemas
│   │
│   ├── infrastructure/           # Shared Agent infrastructure
│   │   ├── persistence/          # Agent runs, conversations and checkpoints
│   │   ├── cache/                # Shared application cache
│   │   ├── security/             # Authentication and authorization
│   │   ├── observability/        # Logging, tracing and metrics
│   │   └── config/               # Application configuration
│   │
│   └── shared/                   # Shared components
│       ├── models/               # Shared domain models
│       ├── exceptions/           # Shared exceptions
│       └── utilities/            # Common utilities
│
├── tests/
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── evaluation/               # Agent quality evaluations
│
├── scripts/                      # Development and maintenance scripts
├── docs/                         # Project documentation
└── deployment/                   # Docker and Kubernetes deployment
```

## Module Overview

| Module           | Responsibility                                                                                |
| ---------------- | --------------------------------------------------------------------------------------------- |
| `core`           | Controls the Agent execution lifecycle, planning, context construction, state and guardrails. |
| `memory`         | Manages short-term and long-term memory, including memory retrieval and storage.              |
| `rag`            | Handles document ingestion, indexing, retrieval, storage and source citations.                |
| `tools`          | Defines, registers, validates and executes tools available to the Agent.                      |
| `integrations`   | Communicates with LLM services, embedding services and other external systems.                |
| `api`            | Exposes REST and streaming interfaces for external clients.                                   |
| `infrastructure` | Provides shared infrastructure used by the Agent application.                                 |
| `shared`         | Contains models, exceptions and utilities shared across multiple modules.                     |
| `tests`          | Contains unit tests, integration tests and Agent evaluations.                                 |

## Module Boundaries

### Core

The `core` module contains the essential Agent execution logic. It coordinates the execution lifecycle but should not directly implement database access, HTTP clients or external service integrations.

### Memory

The `memory` module owns all memory-related behavior and storage.

This includes:

* deciding what information should be remembered;
* retrieving relevant memories;
* managing short-term and long-term memory;
* persisting memory data.

Memory-specific persistence belongs in:

```text
memory/storage/
```

It should not be placed in the shared `infrastructure/persistence` module.

### RAG

The `rag` module owns the complete Retrieval-Augmented Generation pipeline.

This includes:

* document ingestion;
* document processing and chunking;
* index creation;
* retrieval and reranking;
* vector and document storage;
* citation generation.

RAG-specific storage belongs in:

```text
rag/storage/
```

### Tools

The `tools` module manages the tools available to the Agent.

It separates tool definitions from registration, validation and execution so that new tools can be added without changing the Agent runtime.

### Integrations

The `integrations` module contains clients for external systems.

For example:

* LLM inference services;
* embedding services;
* search services;
* internal company APIs;
* third-party APIs.

This module handles communication with external systems but does not define Agent execution behavior.

### API

The `api` module exposes the Agent to external clients.

It is responsible for:

* HTTP endpoints;
* streaming responses;
* request and response schemas;
* middleware;
* transport-level error handling.

The API layer delegates Agent execution to the `core` module.

### Infrastructure

The `infrastructure` module contains infrastructure shared by the Agent application as a whole.

For example:

* Agent run persistence;
* conversation persistence;
* execution checkpoints;
* shared caching;
* authentication and authorization;
* logging, tracing and metrics;
* application configuration.

Domain-specific persistence should remain inside the corresponding module:

```text
memory/storage/
rag/storage/
```

The shared infrastructure layer should only contain storage and technical services that belong to the Agent application itself.

## Design Principles

* Organize the project by responsibility and capability.
* Keep the Agent execution logic inside `core`.
* Treat Memory, RAG and Tools as independent first-class modules.
* Keep domain-specific storage inside the corresponding domain module.
* Use `infrastructure` only for shared Agent application infrastructure.
* Isolate external service communication inside `integrations`.
* Keep the API layer independent from the Agent execution implementation.
* Avoid placing module-specific models and utilities inside `shared`.
* Design module boundaries so that individual capabilities can later be extracted into separate services.
