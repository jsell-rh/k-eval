"""AgentObserver port — domain events emitted during agent invocations."""

from typing import Protocol


class AgentObserver(Protocol):
    """Observer port for agent domain events.

    Implementations may log to structlog, record for tests, or emit metrics.
    """

    def agent_invocation_started(
        self, condition: str, sample_idx: str, model: str
    ) -> None: ...

    def agent_invocation_completed(
        self,
        condition: str,
        sample_idx: str,
        duration_ms: int,
        num_turns: int,
        cost_usd: float | None,
    ) -> None: ...

    def agent_invocation_failed(
        self, condition: str, sample_idx: str, reason: str
    ) -> None: ...
