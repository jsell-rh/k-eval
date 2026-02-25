"""StructlogEvaluationObserver — production observer that delegates to structlog."""

import structlog


class StructlogEvaluationObserver:
    """Logs evaluation domain events to structlog.

    Does NOT inherit from EvaluationObserver (structural typing via Protocol).
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger()

    def evaluation_started(
        self,
        run_id: str,
        total_samples: int,
        total_conditions: int,
        num_samples: int,
        max_concurrent: int,
    ) -> None:
        self._log.info(
            "evaluation.started",
            run_id=run_id,
            total_samples=total_samples,
            total_conditions=total_conditions,
            num_samples=num_samples,
            max_concurrent=max_concurrent,
        )

    def evaluation_completed(self, run_id: str, total_runs: int) -> None:
        self._log.info(
            "evaluation.completed",
            run_id=run_id,
            total_runs=total_runs,
        )

    def sample_condition_started(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        run_index: int,
    ) -> None:
        self._log.info(
            "evaluation.sample_condition.started",
            run_id=run_id,
            sample_idx=sample_idx,
            condition=condition,
            run_index=run_index,
        )

    def sample_condition_completed(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        run_index: int,
    ) -> None:
        self._log.info(
            "evaluation.sample_condition.completed",
            run_id=run_id,
            sample_idx=sample_idx,
            condition=condition,
            run_index=run_index,
        )

    def sample_condition_failed(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        run_index: int,
        reason: str,
    ) -> None:
        self._log.error(
            "evaluation.sample_condition.failed",
            run_id=run_id,
            sample_idx=sample_idx,
            condition=condition,
            run_index=run_index,
            reason=reason,
        )

    def sample_condition_retry(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        run_index: int,
        attempt: int,
        reason: str,
        backoff_seconds: float,
    ) -> None:
        self._log.warning(
            "evaluation.sample_condition.retry",
            run_id=run_id,
            sample_idx=sample_idx,
            condition=condition,
            run_index=run_index,
            attempt=attempt,
            reason=reason,
            backoff_seconds=backoff_seconds,
        )
