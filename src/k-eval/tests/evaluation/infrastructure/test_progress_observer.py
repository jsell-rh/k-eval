"""Tests for ProgressEvaluationObserver (Rich-based, disabled=True for state checks)."""

from evaluation.infrastructure.progress_observer import ProgressEvaluationObserver


def _make_observer() -> ProgressEvaluationObserver:
    return ProgressEvaluationObserver(disabled=True)


def _start(
    observer: ProgressEvaluationObserver,
    condition_names: list[str],
    total_samples: int = 3,
    num_repetitions: int = 1,
) -> None:
    observer.evaluation_started(
        run_id="r",
        total_samples=total_samples,
        total_conditions=len(condition_names),
        condition_names=condition_names,
        num_repetitions=num_repetitions,
        max_concurrent=4,
    )


class TestProgressEvaluationObserverStarted:
    """After evaluation_started, totals are set and done/inflight are zeroed."""

    def test_total_per_condition_is_samples_times_reps(self) -> None:
        observer = _make_observer()
        _start(
            observer=observer,
            condition_names=["cond-a", "cond-b"],
            total_samples=3,
            num_repetitions=2,
        )
        assert observer._total["cond-a"] == 6
        assert observer._total["cond-b"] == 6

    def test_overall_total_is_samples_times_conditions_times_reps(self) -> None:
        observer = _make_observer()
        _start(
            observer=observer,
            condition_names=["cond-a", "cond-b"],
            total_samples=3,
            num_repetitions=2,
        )
        # 3 samples * 2 conditions * 2 reps = 12
        assert observer._total["Overall"] == 12

    def test_done_is_zero_after_started(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["cond-a", "cond-b"])
        assert observer._done["cond-a"] == 0
        assert observer._done["cond-b"] == 0
        assert observer._done["Overall"] == 0

    def test_inflight_is_zero_after_started(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["cond-a", "cond-b"])
        assert observer._inflight["cond-a"] == 0
        assert observer._inflight["cond-b"] == 0
        assert observer._inflight["Overall"] == 0

    def test_all_conditions_present_in_total(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["alpha", "beta", "gamma"])
        assert set(observer._total.keys()) == {"alpha", "beta", "gamma", "Overall"}

    def test_no_state_before_started(self) -> None:
        observer = _make_observer()
        assert observer._total == {}
        assert observer._done == {}
        assert observer._inflight == {}


class TestProgressEvaluationObserverProgress:
    """evaluation_progress increments done and decrements inflight for condition and Overall."""

    def test_single_progress_increments_done(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["baseline"], total_samples=5)
        observer.evaluation_progress(
            run_id="r", condition="baseline", completed=1, total=5
        )
        assert observer._done["baseline"] == 1
        assert observer._done["Overall"] == 1

    def test_multiple_progress_events_accumulate(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["baseline"], total_samples=5)
        for i in range(1, 4):
            observer.evaluation_progress(
                run_id="r", condition="baseline", completed=i, total=5
            )
        assert observer._done["baseline"] == 3
        assert observer._done["Overall"] == 3

    def test_two_conditions_routed_separately(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["alpha", "beta"], total_samples=2)
        observer.evaluation_progress(
            run_id="r", condition="alpha", completed=1, total=4
        )
        observer.evaluation_progress(run_id="r", condition="beta", completed=1, total=4)
        observer.evaluation_progress(
            run_id="r", condition="alpha", completed=2, total=4
        )
        assert observer._done["alpha"] == 2
        assert observer._done["beta"] == 1
        assert observer._done["Overall"] == 3

    def test_progress_decrements_inflight(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["baseline"], total_samples=5)
        # Simulate: start one, then complete it via progress.
        observer.sample_condition_started(
            run_id="r", sample_idx="s0", condition="baseline", repetition_index=0
        )
        assert observer._inflight["baseline"] == 1
        observer.evaluation_progress(
            run_id="r", condition="baseline", completed=1, total=5
        )
        assert observer._inflight["baseline"] == 0
        assert observer._inflight["Overall"] == 0

    def test_inflight_does_not_go_below_zero(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["baseline"], total_samples=5)
        # Progress without a preceding sample_condition_started (inflight already 0).
        observer.evaluation_progress(
            run_id="r", condition="baseline", completed=1, total=5
        )
        assert observer._inflight["baseline"] == 0

    def test_progress_before_started_is_noop(self) -> None:
        observer = _make_observer()
        observer.evaluation_progress(
            run_id="r", condition="baseline", completed=1, total=5
        )
        assert observer._done == {}


class TestProgressEvaluationObserverInFlight:
    """sample_condition_started increments inflight; evaluation_progress decrements it."""

    def test_sample_condition_started_increments_inflight(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["cond-a"])
        observer.sample_condition_started(
            run_id="r", sample_idx="s0", condition="cond-a", repetition_index=0
        )
        assert observer._inflight["cond-a"] == 1
        assert observer._inflight["Overall"] == 1

    def test_multiple_starts_accumulate_inflight(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["cond-a"], total_samples=5)
        for i in range(3):
            observer.sample_condition_started(
                run_id="r", sample_idx=f"s{i}", condition="cond-a", repetition_index=0
            )
        assert observer._inflight["cond-a"] == 3
        assert observer._inflight["Overall"] == 3

    def test_inflight_decrements_on_evaluation_progress(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["cond-a"], total_samples=5)
        observer.sample_condition_started(
            run_id="r", sample_idx="s0", condition="cond-a", repetition_index=0
        )
        observer.sample_condition_started(
            run_id="r", sample_idx="s1", condition="cond-a", repetition_index=0
        )
        assert observer._inflight["cond-a"] == 2
        observer.evaluation_progress(
            run_id="r", condition="cond-a", completed=1, total=5
        )
        assert observer._inflight["cond-a"] == 1
        assert observer._inflight["Overall"] == 1


class TestProgressEvaluationObserverCompleted:
    """evaluation_completed clears all internal state."""

    def test_state_cleared_after_completed(self) -> None:
        observer = _make_observer()
        _start(observer=observer, condition_names=["cond-a", "cond-b"])
        observer.evaluation_completed(run_id="r", total_runs=4, elapsed_seconds=1.5)
        assert observer._done == {}
        assert observer._inflight == {}
        assert observer._total == {}
        assert observer._task_ids == {}
        assert observer._overall_progress is None
        assert observer._condition_progress is None
        assert observer._live is None

    def test_completed_before_started_does_not_raise(self) -> None:
        observer = _make_observer()
        observer.evaluation_completed(run_id="r", total_runs=0, elapsed_seconds=0.0)


class TestProgressEvaluationObserverNoOps:
    """sample_condition_completed, sample_condition_failed, and retry do not affect done."""

    def _started_observer(self) -> ProgressEvaluationObserver:
        observer = _make_observer()
        _start(observer=observer, condition_names=["cond-a"])
        return observer

    def test_sample_condition_completed_does_not_change_done(self) -> None:
        observer = self._started_observer()
        observer.sample_condition_completed(
            run_id="r", sample_idx="s0", condition="cond-a", repetition_index=0
        )
        assert observer._done["cond-a"] == 0
        assert observer._done["Overall"] == 0

    def test_sample_condition_failed_does_not_change_done(self) -> None:
        observer = self._started_observer()
        observer.sample_condition_failed(
            run_id="r",
            sample_idx="s0",
            condition="cond-a",
            repetition_index=0,
            reason="bad",
        )
        assert observer._done["cond-a"] == 0
        assert observer._done["Overall"] == 0

    def test_sample_condition_retry_does_not_change_done_or_inflight(self) -> None:
        observer = self._started_observer()
        observer.sample_condition_retry(
            run_id="r",
            sample_idx="s0",
            condition="cond-a",
            repetition_index=0,
            attempt=1,
            reason="timeout",
            backoff_seconds=1.0,
        )
        assert observer._done["cond-a"] == 0
        assert observer._inflight["cond-a"] == 0

    def test_sample_condition_completed_does_not_affect_inflight(self) -> None:
        observer = self._started_observer()
        observer.sample_condition_started(
            run_id="r", sample_idx="s0", condition="cond-a", repetition_index=0
        )
        assert observer._inflight["cond-a"] == 1
        observer.sample_condition_completed(
            run_id="r", sample_idx="s0", condition="cond-a", repetition_index=0
        )
        # inflight unchanged — only evaluation_progress decrements it
        assert observer._inflight["cond-a"] == 1


class TestProgressEvaluationObserverRetry:
    """Retry edge cases: in-flight must not leak across retry attempts."""

    def _started_observer(self) -> ProgressEvaluationObserver:
        observer = _make_observer()
        _start(observer=observer, condition_names=["cond-a"])
        return observer

    def test_retry_decrements_inflight_during_backoff(self) -> None:
        observer = self._started_observer()
        observer.sample_condition_started(
            run_id="r", sample_idx="s0", condition="cond-a", repetition_index=0
        )
        assert observer._inflight["cond-a"] == 1

        # Triple fails and enters backoff — inflight should drop to 0.
        observer.sample_condition_retry(
            run_id="r",
            sample_idx="s0",
            condition="cond-a",
            repetition_index=0,
            attempt=1,
            reason="timeout",
            backoff_seconds=1.0,
        )
        assert observer._inflight["cond-a"] == 0
        assert observer._inflight["Overall"] == 0

    def test_retry_then_progress_does_not_go_negative(self) -> None:
        observer = self._started_observer()
        observer.sample_condition_started(
            run_id="r", sample_idx="s0", condition="cond-a", repetition_index=0
        )
        observer.sample_condition_retry(
            run_id="r",
            sample_idx="s0",
            condition="cond-a",
            repetition_index=0,
            attempt=1,
            reason="timeout",
            backoff_seconds=1.0,
        )
        # Re-enters sem on attempt 2 (runner does NOT re-emit started).
        # Then succeeds — evaluation_progress decrements inflight.
        observer.evaluation_progress(
            run_id="r", condition="cond-a", completed=1, total=3
        )
        assert observer._inflight["cond-a"] == 0
        assert observer._done["cond-a"] == 1

    def test_multiple_retries_do_not_leak_inflight(self) -> None:
        observer = self._started_observer()
        observer.sample_condition_started(
            run_id="r", sample_idx="s0", condition="cond-a", repetition_index=0
        )
        # Two retries — each backoff should decrement inflight.
        for attempt in range(1, 3):
            observer.sample_condition_retry(
                run_id="r",
                sample_idx="s0",
                condition="cond-a",
                repetition_index=0,
                attempt=attempt,
                reason="timeout",
                backoff_seconds=1.0,
            )
        # inflight floored at 0, never goes negative.
        assert observer._inflight["cond-a"] == 0
        assert observer._inflight["Overall"] == 0
