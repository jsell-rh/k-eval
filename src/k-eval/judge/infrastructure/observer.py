"""Structlog implementation of the JudgeObserver port."""

import structlog


class StructlogJudgeObserver:
    """Delegates judge domain events to structlog.

    Satisfies the JudgeObserver protocol structurally.
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger()

    def judge_scoring_started(
        self, condition: str, sample_idx: str, model: str
    ) -> None:
        self._log.info(
            "judge.scoring_started",
            condition=condition,
            sample_idx=sample_idx,
            model=model,
        )

    def judge_scoring_completed(
        self, condition: str, sample_idx: str, duration_ms: int
    ) -> None:
        self._log.info(
            "judge.scoring_completed",
            condition=condition,
            sample_idx=sample_idx,
            duration_ms=duration_ms,
        )

    def judge_scoring_failed(
        self, condition: str, sample_idx: str, reason: str
    ) -> None:
        self._log.error(
            "judge.scoring_failed",
            condition=condition,
            sample_idx=sample_idx,
            reason=reason,
        )

    def judge_high_temperature_warned(
        self, condition: str, sample_idx: str, temperature: float
    ) -> None:
        self._log.warning(
            "judge.high_temperature_warned",
            condition=condition,
            sample_idx=sample_idx,
            temperature=temperature,
        )
