"""Observer port for the evaluation domain — defines events in domain language."""

from typing import Protocol


class EvaluationObserver(Protocol):
    """Observer port emitting structured events during an evaluation run.

    Implementations may log to structlog, record for tests, or emit metrics.
    """

    def evaluation_started(
        self,
        run_id: str,
        total_samples: int,
        total_conditions: int,
        num_samples: int,
        max_concurrent: int,
    ) -> None: ...

    def evaluation_completed(self, run_id: str, total_runs: int) -> None: ...

    def sample_condition_started(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        run_index: int,
    ) -> None: ...

    def sample_condition_completed(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        run_index: int,
    ) -> None: ...

    def sample_condition_failed(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        run_index: int,
        reason: str,
    ) -> None: ...

    def sample_condition_retry(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        run_index: int,
        attempt: int,
        reason: str,
        backoff_seconds: float,
    ) -> None: ...
