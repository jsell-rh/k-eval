"""AgentResult value object — the outcome of a single agent invocation."""

from dataclasses import dataclass

from agent.domain.usage import UsageMetrics


@dataclass(frozen=True)
class AgentResult:
    """Immutable value object capturing all outcome data from one agent invocation."""

    response: str
    cost_usd: float | None
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    usage: UsageMetrics | None
