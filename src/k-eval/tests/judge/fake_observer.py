"""FakeJudgeObserver — records judge domain events for assertion in tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringStartedEvent:
    condition: str
    sample_id: str
    model: str


@dataclass(frozen=True)
class ScoringCompletedEvent:
    condition: str
    sample_id: str
    duration_ms: int


@dataclass(frozen=True)
class ScoringFailedEvent:
    condition: str
    sample_id: str
    reason: str


@dataclass(frozen=True)
class HighTemperatureWarnedEvent:
    condition: str
    sample_id: str
    temperature: float


class FakeJudgeObserver:
    """Records all emitted judge events as typed frozen dataclasses.

    Use in tests to assert which events were emitted and with what data,
    without mocking or patching.
    """

    def __init__(self) -> None:
        self.started: list[ScoringStartedEvent] = []
        self.completed: list[ScoringCompletedEvent] = []
        self.failed: list[ScoringFailedEvent] = []
        self.temperature_warnings: list[HighTemperatureWarnedEvent] = []

    def judge_scoring_started(self, condition: str, sample_id: str, model: str) -> None:
        self.started.append(
            ScoringStartedEvent(condition=condition, sample_id=sample_id, model=model)
        )

    def judge_scoring_completed(
        self, condition: str, sample_id: str, duration_ms: int
    ) -> None:
        self.completed.append(
            ScoringCompletedEvent(
                condition=condition, sample_id=sample_id, duration_ms=duration_ms
            )
        )

    def judge_scoring_failed(self, condition: str, sample_id: str, reason: str) -> None:
        self.failed.append(
            ScoringFailedEvent(condition=condition, sample_id=sample_id, reason=reason)
        )

    def judge_high_temperature_warned(
        self, condition: str, sample_id: str, temperature: float
    ) -> None:
        self.temperature_warnings.append(
            HighTemperatureWarnedEvent(
                condition=condition, sample_id=sample_id, temperature=temperature
            )
        )
