# Project Context for AI Coding Assistants

## Project Overview

**Project Name:** k-eval

**Description:** Context-aware evaluation framework for AI agents using MCP.

**Tech Stack:**
- Language: Python 3.13
- Key Dependencies: Pydantic, Pydantic Settings, pyyaml, litellm, claude-agent-sdk, structlog, uv

## Architecture

### Domain Driven Design (DDD)

k-eval is organized domain-first, then by layer within each domain. Each top-level directory represents a domain concept. Within it, code is split into layers:

- **domain/**: Core concepts, value objects, aggregates, and domain services. No infrastructure dependencies.
- **application/**: Orchestrates domain objects to fulfill use cases (e.g. running an evaluation).
- **infrastructure/**: Implements interfaces defined by the domain (agent SDK, LiteLLM, file I/O, YAML loading).

**Example structure:**
```
evaluation/
    domain/
        run.py
        condition.py
        observer.py     # EvaluationObserver Protocol (port)
    application/
        runner.py
    infrastructure/
        yaml_loader.py
        observer.py     # StructlogEvaluationObserver implementation
agent/
    domain/
        agent.py        # Agent Protocol definition
        result.py
        observer.py     # AgentObserver Protocol (port)
    infrastructure/
        claude_sdk.py   # ClaudeAgentSDKAgent implementation
        observer.py     # StructlogAgentObserver implementation
judge/
    domain/
        judge.py        # Judge Protocol definition
        score.py
        observer.py     # JudgeObserver Protocol (port)
    infrastructure/
        litellm.py      # LiteLLMJudge implementation
        observer.py     # StructlogJudgeObserver implementation
dataset/
    domain/
        sample.py
        observer.py     # DatasetObserver Protocol (port)
    infrastructure/
        jsonl_loader.py
        observer.py     # StructlogDatasetObserver implementation
```

Domain objects must not import from infrastructure. Infrastructure imports from domain. Application imports from both.

### Module Structure

Modules are atomic: one concept per file, named for that concept. Prefer a directory of focused modules over a single large file.

**Correct:**
```
agent/infrastructure/claude_sdk.py
agent/infrastructure/base.py
```

**Incorrect:**
```
agent/infrastructure/agents.py  # too broad if it contains multiple agent types
```

This applies at every layer within every domain. If a module grows to contain more than one primary concept, split it.

### Domain-Oriented Observability (DOO)

Observability is expressed in domain terms, not infrastructure terms. Do not call `logger.info(...)` directly in domain or application code. Instead, emit structured domain events through 
**Observer** objects injected into domain services via a defined **Port** (Protocol).

This pattern enables swapping implementations: production uses structlog; tests use an in-memory fake that records calls for assertion without mocking.

**Port** — defined in `<domain>/domain/observer.py`, expresses what events exist in domain language:

```python
# evaluation/domain/observer.py
from typing import Protocol

class EvaluationObserver(Protocol):
    def evaluation_started(self, run_id: str, condition: str, sample_id: str) -> None: ...
    def evaluation_failed(self, run_id: str, condition: str, sample_id: str, reason: str) -> None: ...
```

**Production implementation** — defined in `<domain>/infrastructure/observer.py`, delegates to structlog:

```python
# evaluation/infrastructure/observer.py
import structlog

class StructlogEvaluationObserver:
    def __init__(self) -> None:
        self._log = structlog.get_logger()

    def evaluation_started(self, run_id: str, condition: str, sample_id: str) -> None:
        self._log.info("evaluation.started", run_id=run_id, condition=condition, sample_id=sample_id)

    def evaluation_failed(self, run_id: str, condition: str, sample_id: str, reason: str) -> None:
        self._log.error("evaluation.failed", run_id=run_id, condition=condition, sample_id=sample_id, reason=reason)
```

**Test fake** — defined in `tests/<domain>/`, records events for assertion without mocking:

```python
# tests/evaluation/fake_observer.py
class FakeEvaluationObserver:
    def __init__(self) -> None:
        self.started: list[dict[str, str]] = []
        self.failed: list[dict[str, str]] = []

    def evaluation_started(self, run_id: str, condition: str, sample_id: str) -> None:
        self.started.append({"run_id": run_id, "condition": condition, "sample_id": sample_id})

    def evaluation_failed(self, run_id: str, condition: str, sample_id: str, reason: str) -> None:
        self.failed.append({"run_id": run_id, "condition": condition, "sample_id": sample_id, "reason": reason})
```

**Domain service** — receives the observer via constructor injection:

```python
# evaluation/application/runner.py
from evaluation.domain.observer import EvaluationObserver

class EvaluationRunner:
    def __init__(self, observer: EvaluationObserver) -> None:
        self._observer = observer
```

Rules:
- Observer Protocols (ports) live in `<domain>/domain/observer.py`
- Production implementations live in `<domain>/infrastructure/observer.py`
- Test fakes live in `tests/<domain>/fake_observer.py`
- Method names are verbs describing domain events (`evaluation_started`, `judge_scored`, `retry_attempted`)
- Methods always take typed, named parameters — never raw dicts or `**kwargs`
- Infrastructure code may call structlog directly only for infrastructure-level events (e.g. HTTP connection errors), not domain events

## Coding Guidelines

### Python 3.13 Type Annotations

- Use `X | Y` union syntax, never `Optional[X]` or `Union[X, Y]`
- Use `type` statement for type aliases: `type SampleId = str`
- Use `Self` from `typing` where appropriate for builder/fluent patterns
- Use `typing.Protocol` to define structural interfaces (e.g. `Agent`, `Judge`)
- Prefer `dataclass` with `frozen=True` for immutable value objects
- Use `StrEnum` for all enumerations of string values
- Never use bare `dict`, `tuple`, or `list` as return or parameter types — always parameterize or define a named type

### Style and Conventions

- Always use type hinting on all function signatures and class attributes
- Prefer `StrEnum`, frozen `dataclass`, or Pydantic models over generic container types
- Use Pydantic models for all config and data transfer objects that cross layer boundaries
- Use Pydantic Settings for runtime configuration (env vars, CLI flags)
- Always use keyword arguments at call sites: `foo(a=1, b=2)` not `foo(1, 2)`. This makes call sites self-documenting and prevents positional argument bugs as signatures evolve.

### Simplicity in Tension with DDD and DOO

Simplicity is a core design principle, but it must be held in tension with DDD and DOO — not used as an excuse to collapse boundaries or skip observability.

- If the simple solution violates a layer boundary, fix the boundary violation
- If the simple solution omits an observer event, add it
- If a concept is genuinely simple and doesn't warrant its own domain object, a well-typed primitive is fine — but name it with a `type` alias so intent is clear
- When in doubt, prefer explicitness over cleverness

### Best Practices

- Always follow TDD: write a failing test before writing implementation
- Keep functions small and focused (single responsibility)
- Add docstrings for complex logic; skip them for obvious code
- Avoid placeholder comments like "next step" or "do the thing"
- Catch specific exception types, never bare `except` or `except Exception`
- Error messages must start with "Failed to ..."
- Extract repeated code into named helpers; follow DRY strictly

## Testing

### Philosophy

Tests must be meaningful and test behavior, not implementation. Minimize mocks — prefer real objects, in-memory fakes, and test fixtures over mock frameworks. A test that only verifies that a mock was called is not a useful test.

- Test domain logic with real domain objects, no mocks
- Test application logic with fakes (in-memory implementations of infrastructure interfaces), not mocks
- Reserve mocks only for external I/O that cannot be faked (e.g. actual LLM API calls in integration tests)
- Extract shared test setup into clearly named helper functions or fixtures; never duplicate setup code across tests

### TDD Workflow

1. Write a failing test that describes the behavior
2. Confirm it fails for the right reason
3. Write the minimal implementation to make it pass
4. Refactor — the test suite is your safety net

### Structure

Tests mirror the source layout. A test for `evaluation/domain/run.py` lives at `tests/evaluation/domain/test_run.py`.

## Common Tasks

### Adding a New Feature
1. Identify which layer the concept belongs to
2. Review existing similar features for patterns
3. Write a failing test
4. Implement the feature
5. Ensure all tests pass

### Adding a New Agent Type
1. Define or verify the `Agent` protocol in `agent/domain/agent.py`
2. Create `agent/infrastructure/<name>.py` with the new implementation
3. Write tests using a fake or recorded fixture — not a live API call
4. Register the new type in config loading

### Fixing a Bug
1. Write a test that reproduces the bug
2. Fix the bug
3. Verify the test passes
4. Check for similar bugs elsewhere

## Important Constraints

- **Security:** Never commit secrets, API keys, or credentials
- **Licensing:** Only use dependencies with compatible licenses
- **Git:** Always use conventional commits. Never skip verify when committing or pushing.

## Notes for AI Assistants

- This project uses `uv` for package management. Always use `uv run ...` when running python, pytest, mypy, or any other tool.
- Never add infrastructure imports to domain layer modules — flag it as a boundary violation instead.
- When adding observability, add an observer method to the port first, then call it — never inline a logger call in domain or application code.
- If a module feels too large, it probably needs to be split. Prefer the atomic module structure.
