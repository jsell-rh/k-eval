"""FakeAgentObserver — records agent domain events for assertion in tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InvocationStartedEvent:
    condition: str
    sample_id: str
    model: str


@dataclass(frozen=True)
class InvocationCompletedEvent:
    condition: str
    sample_id: str
    duration_ms: int
    num_turns: int
    cost_usd: float | None


@dataclass(frozen=True)
class InvocationFailedEvent:
    condition: str
    sample_id: str
    reason: str


class FakeAgentObserver:
    """Records all emitted agent events as typed frozen dataclasses.

    Use in tests to assert which events were emitted and with what data,
    without mocking or patching.
    """

    def __init__(self) -> None:
        self.invocation_started: list[InvocationStartedEvent] = []
        self.invocation_completed: list[InvocationCompletedEvent] = []
        self.invocation_failed: list[InvocationFailedEvent] = []

    def agent_invocation_started(
        self, condition: str, sample_id: str, model: str
    ) -> None:
        self.invocation_started.append(
            InvocationStartedEvent(
                condition=condition, sample_id=sample_id, model=model
            )
        )

    def agent_invocation_completed(
        self,
        condition: str,
        sample_id: str,
        duration_ms: int,
        num_turns: int,
        cost_usd: float | None,
    ) -> None:
        self.invocation_completed.append(
            InvocationCompletedEvent(
                condition=condition,
                sample_id=sample_id,
                duration_ms=duration_ms,
                num_turns=num_turns,
                cost_usd=cost_usd,
            )
        )

    def agent_invocation_failed(
        self, condition: str, sample_id: str, reason: str
    ) -> None:
        self.invocation_failed.append(
            InvocationFailedEvent(
                condition=condition, sample_id=sample_id, reason=reason
            )
        )
