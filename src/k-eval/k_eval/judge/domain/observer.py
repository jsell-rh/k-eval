"""JudgeObserver port — domain events emitted during judge invocations."""

from typing import Protocol


class JudgeObserver(Protocol):
    """Observer port for judge domain events.

    Implementations may log to structlog, record for tests, or emit metrics.
    """

    def judge_scoring_started(
        self, condition: str, sample_idx: str, model: str
    ) -> None: ...

    def judge_scoring_completed(
        self, condition: str, sample_idx: str, duration_ms: int
    ) -> None: ...

    def judge_scoring_failed(
        self, condition: str, sample_idx: str, reason: str
    ) -> None: ...

    def judge_high_temperature_warned(
        self, condition: str, sample_idx: str, temperature: float
    ) -> None: ...
