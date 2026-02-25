"""ProgressEvaluationObserver — writes a live progress bar to stderr."""

import sys


class ProgressEvaluationObserver:
    """Renders a live progress bar on stderr during an evaluation run.

    All other events are no-ops; only evaluation_progress and
    evaluation_completed are rendered.

    Does NOT inherit from EvaluationObserver (structural typing via Protocol).
    """

    _BAR_WIDTH = 30

    def evaluation_started(
        self,
        run_id: str,
        total_samples: int,
        total_conditions: int,
        num_repetitions: int,
        max_concurrent: int,
    ) -> None:
        pass

    def evaluation_completed(
        self,
        run_id: str,
        total_runs: int,
        elapsed_seconds: float,
    ) -> None:
        # Overwrite the progress line with a clean completion message.
        minutes, seconds = divmod(elapsed_seconds, 60)
        if minutes >= 1:
            time_str = f"{int(minutes)}m {seconds:.1f}s"
        else:
            time_str = f"{elapsed_seconds:.1f}s"
        sys.stderr.write(f"\r\033[K✓ {total_runs} triples completed in {time_str}\n")
        sys.stderr.flush()

    def evaluation_progress(
        self,
        run_id: str,
        completed: int,
        total: int,
    ) -> None:
        fraction = completed / total if total else 0.0
        filled = round(fraction * self._BAR_WIDTH)
        bar = "█" * filled + "░" * (self._BAR_WIDTH - filled)
        percent = fraction * 100.0
        # \r returns to line start; \033[K clears to end of line.
        sys.stderr.write(f"\r\033[K  [{bar}] {completed}/{total}  {percent:.0f}%")
        sys.stderr.flush()

    def sample_condition_started(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        repetition_index: int,
    ) -> None:
        pass

    def sample_condition_completed(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        repetition_index: int,
    ) -> None:
        pass

    def sample_condition_failed(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        repetition_index: int,
        reason: str,
    ) -> None:
        pass

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
        pass
