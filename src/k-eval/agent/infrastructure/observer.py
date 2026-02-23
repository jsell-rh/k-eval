"""Structlog implementation of the AgentObserver port."""

import structlog


class StructlogAgentObserver:
    """Delegates agent domain events to structlog.

    Satisfies the AgentObserver protocol structurally.
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger()

    def agent_invocation_started(
        self, condition: str, sample_id: str, model: str
    ) -> None:
        self._log.info(
            "agent.invocation_started",
            condition=condition,
            sample_id=sample_id,
            model=model,
        )

    def agent_invocation_completed(
        self,
        condition: str,
        sample_id: str,
        duration_ms: int,
        num_turns: int,
        cost_usd: float | None,
    ) -> None:
        self._log.info(
            "agent.invocation_completed",
            condition=condition,
            sample_id=sample_id,
            duration_ms=duration_ms,
            num_turns=num_turns,
            cost_usd=cost_usd,
        )

    def agent_invocation_failed(
        self, condition: str, sample_id: str, reason: str
    ) -> None:
        self._log.error(
            "agent.invocation_failed",
            condition=condition,
            sample_id=sample_id,
            reason=reason,
        )
