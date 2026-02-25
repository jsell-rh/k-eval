"""FakeEvaluationObserver — records evaluation domain events for assertion in tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationStartedEvent:
    run_id: str
    total_samples: int
    total_conditions: int
    condition_names: list[str]
    num_repetitions: int
    max_concurrent: int


@dataclass(frozen=True)
class EvaluationCompletedEvent:
    run_id: str
    total_runs: int
    elapsed_seconds: float


@dataclass(frozen=True)
class EvaluationProgressEvent:
    run_id: str
    condition: str
    completed: int
    total: int


@dataclass(frozen=True)
class SampleConditionStartedEvent:
    run_id: str
    sample_idx: str
    condition: str
    repetition_index: int


@dataclass(frozen=True)
class SampleConditionCompletedEvent:
    run_id: str
    sample_idx: str
    condition: str
    repetition_index: int


@dataclass(frozen=True)
class SampleConditionFailedEvent:
    run_id: str
    sample_idx: str
    condition: str
    repetition_index: int
    reason: str


@dataclass(frozen=True)
class SampleConditionRetryEvent:
    run_id: str
    sample_idx: str
    condition: str
    repetition_index: int
    attempt: int
    reason: str
    backoff_seconds: float


class FakeEvaluationObserver:
    """Records all emitted evaluation events as typed frozen dataclasses.

    Use in tests to assert which events were emitted and with what data,
    without mocking or patching.

    Event lists use a leading underscore + public property pattern to avoid
    name collision between the list attributes and the Protocol method names.
    """

    def __init__(self) -> None:
        self._started: list[EvaluationStartedEvent] = []
        self._completed: list[EvaluationCompletedEvent] = []
        self._progress: list[EvaluationProgressEvent] = []
        self._sc_started: list[SampleConditionStartedEvent] = []
        self._sc_completed: list[SampleConditionCompletedEvent] = []
        self._sc_failed: list[SampleConditionFailedEvent] = []
        self._sc_retried: list[SampleConditionRetryEvent] = []

    @property
    def started(self) -> list[EvaluationStartedEvent]:
        return self._started

    @property
    def completed(self) -> list[EvaluationCompletedEvent]:
        return self._completed

    @property
    def progress(self) -> list[EvaluationProgressEvent]:
        return self._progress

    @property
    def sc_started(self) -> list[SampleConditionStartedEvent]:
        return self._sc_started

    @property
    def sc_completed(self) -> list[SampleConditionCompletedEvent]:
        return self._sc_completed

    @property
    def sc_failed(self) -> list[SampleConditionFailedEvent]:
        return self._sc_failed

    @property
    def sc_retried(self) -> list[SampleConditionRetryEvent]:
        return self._sc_retried

    def evaluation_started(
        self,
        run_id: str,
        total_samples: int,
        total_conditions: int,
        condition_names: list[str],
        num_repetitions: int,
        max_concurrent: int,
    ) -> None:
        self._started.append(
            EvaluationStartedEvent(
                run_id=run_id,
                total_samples=total_samples,
                total_conditions=total_conditions,
                condition_names=condition_names,
                num_repetitions=num_repetitions,
                max_concurrent=max_concurrent,
            )
        )

    def evaluation_completed(
        self,
        run_id: str,
        total_runs: int,
        elapsed_seconds: float,
    ) -> None:
        self._completed.append(
            EvaluationCompletedEvent(
                run_id=run_id,
                total_runs=total_runs,
                elapsed_seconds=elapsed_seconds,
            )
        )

    def evaluation_progress(
        self,
        run_id: str,
        condition: str,
        completed: int,
        total: int,
    ) -> None:
        self._progress.append(
            EvaluationProgressEvent(
                run_id=run_id,
                condition=condition,
                completed=completed,
                total=total,
            )
        )

    def sample_condition_started(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        repetition_index: int,
    ) -> None:
        self._sc_started.append(
            SampleConditionStartedEvent(
                run_id=run_id,
                sample_idx=sample_idx,
                condition=condition,
                repetition_index=repetition_index,
            )
        )

    def sample_condition_completed(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        repetition_index: int,
    ) -> None:
        self._sc_completed.append(
            SampleConditionCompletedEvent(
                run_id=run_id,
                sample_idx=sample_idx,
                condition=condition,
                repetition_index=repetition_index,
            )
        )

    def sample_condition_failed(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        repetition_index: int,
        reason: str,
    ) -> None:
        self._sc_failed.append(
            SampleConditionFailedEvent(
                run_id=run_id,
                sample_idx=sample_idx,
                condition=condition,
                repetition_index=repetition_index,
                reason=reason,
            )
        )

    def sample_condition_retry(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        repetition_index: int,
        attempt: int,
        reason: str,
        backoff_seconds: float,
    ) -> None:
        self._sc_retried.append(
            SampleConditionRetryEvent(
                run_id=run_id,
                sample_idx=sample_idx,
                condition=condition,
                repetition_index=repetition_index,
                attempt=attempt,
                reason=reason,
                backoff_seconds=backoff_seconds,
            )
        )
