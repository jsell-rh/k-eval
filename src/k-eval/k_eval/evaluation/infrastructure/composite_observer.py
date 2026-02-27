"""CompositeEvaluationObserver — fans out all events to a list of observers."""

from k_eval.evaluation.domain.observer import EvaluationObserver


class CompositeEvaluationObserver:
    """Delegates every observer event to each observer in order.

    Does NOT inherit from EvaluationObserver (structural typing via Protocol).
    """

    def __init__(self, observers: list[EvaluationObserver]) -> None:
        self._observers = observers

    def evaluation_started(
        self,
        run_id: str,
        total_samples: int,
        total_conditions: int,
        condition_names: list[str],
        num_repetitions: int,
        max_concurrent: int,
    ) -> None:
        for obs in self._observers:
            obs.evaluation_started(
                run_id=run_id,
                total_samples=total_samples,
                total_conditions=total_conditions,
                condition_names=condition_names,
                num_repetitions=num_repetitions,
                max_concurrent=max_concurrent,
            )

    def evaluation_completed(
        self,
        run_id: str,
        total_runs: int,
        elapsed_seconds: float,
    ) -> None:
        for obs in self._observers:
            obs.evaluation_completed(
                run_id=run_id,
                total_runs=total_runs,
                elapsed_seconds=elapsed_seconds,
            )

    def evaluation_progress(
        self,
        run_id: str,
        condition: str,
        completed: int,
        total: int,
    ) -> None:
        for obs in self._observers:
            obs.evaluation_progress(
                run_id=run_id,
                condition=condition,
                completed=completed,
                total=total,
            )

    def sample_condition_started(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        repetition_index: int,
    ) -> None:
        for obs in self._observers:
            obs.sample_condition_started(
                run_id=run_id,
                sample_idx=sample_idx,
                condition=condition,
                repetition_index=repetition_index,
            )

    def sample_condition_completed(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        repetition_index: int,
    ) -> None:
        for obs in self._observers:
            obs.sample_condition_completed(
                run_id=run_id,
                sample_idx=sample_idx,
                condition=condition,
                repetition_index=repetition_index,
            )

    def sample_condition_failed(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        repetition_index: int,
        reason: str,
    ) -> None:
        for obs in self._observers:
            obs.sample_condition_failed(
                run_id=run_id,
                sample_idx=sample_idx,
                condition=condition,
                repetition_index=repetition_index,
                reason=reason,
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
        for obs in self._observers:
            obs.sample_condition_retry(
                run_id=run_id,
                sample_idx=sample_idx,
                condition=condition,
                repetition_index=repetition_index,
                attempt=attempt,
                reason=reason,
                backoff_seconds=backoff_seconds,
            )

    def mcp_tool_use_absent(
        self,
        run_id: str,
        condition: str,
        sample_idx: int,
        repetition_index: int,
    ) -> None:
        for obs in self._observers:
            obs.mcp_tool_use_absent(
                run_id=run_id,
                condition=condition,
                sample_idx=sample_idx,
                repetition_index=repetition_index,
            )

    def mcp_tool_success_absent(
        self,
        run_id: str,
        condition: str,
        sample_idx: int,
        repetition_index: int,
    ) -> None:
        for obs in self._observers:
            obs.mcp_tool_success_absent(
                run_id=run_id,
                condition=condition,
                sample_idx=sample_idx,
                repetition_index=repetition_index,
            )
