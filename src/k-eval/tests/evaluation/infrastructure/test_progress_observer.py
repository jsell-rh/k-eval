"""Tests for ProgressEvaluationObserver."""

from typing import Any

from evaluation.infrastructure.progress_observer import ProgressEvaluationObserver


class SpyBar:
    """In-memory stand-in for a tqdm bar that records calls."""

    def __init__(self) -> None:
        self.update_calls: list[int] = []
        self.closed = False

    def update(self, n: int = 1) -> None:
        self.update_calls.append(n)

    def close(self) -> None:
        self.closed = True


class SpyTqdmFactory:
    """Factory that produces SpyBar instances and remembers the kwargs used."""

    def __init__(self) -> None:
        self.bar = SpyBar()
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> SpyBar:
        self.kwargs = kwargs
        return self.bar


def _make_observer() -> tuple[ProgressEvaluationObserver, SpyTqdmFactory]:
    factory = SpyTqdmFactory()
    observer = ProgressEvaluationObserver(tqdm_factory=factory)
    return observer, factory


class TestProgressEvaluationObserverStarted:
    """evaluation_started creates a tqdm bar with the correct total."""

    def test_bar_created_on_evaluation_started(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=3,
            total_conditions=2,
            num_repetitions=2,
            max_concurrent=4,
        )
        # total = 3 * 2 * 2 = 12
        assert factory.kwargs["total"] == 12

    def test_bar_unit_is_triple(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=1,
            total_conditions=1,
            num_repetitions=1,
            max_concurrent=1,
        )
        assert factory.kwargs["unit"] == "triple"

    def test_bar_not_created_before_evaluation_started(self) -> None:
        observer, factory = _make_observer()
        # No evaluation_started call — factory should never have been called.
        assert factory.kwargs == {}


class TestProgressEvaluationObserverProgress:
    """evaluation_progress increments the bar by 1 each call."""

    def test_single_progress_event_increments_bar_once(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=5,
            total_conditions=1,
            num_repetitions=1,
            max_concurrent=1,
        )
        observer.evaluation_progress(run_id="r", completed=1, total=5)
        assert factory.bar.update_calls == [1]

    def test_multiple_progress_events_each_increment_by_one(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=3,
            total_conditions=1,
            num_repetitions=1,
            max_concurrent=1,
        )
        for i in range(1, 4):
            observer.evaluation_progress(run_id="r", completed=i, total=3)
        assert factory.bar.update_calls == [1, 1, 1]

    def test_progress_before_started_does_not_raise(self) -> None:
        observer, factory = _make_observer()
        # No evaluation_started — _bar is None; should be a safe no-op.
        observer.evaluation_progress(run_id="r", completed=1, total=5)
        assert factory.bar.update_calls == []


class TestProgressEvaluationObserverCompleted:
    """evaluation_completed closes the bar."""

    def test_bar_closed_on_evaluation_completed(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=2,
            total_conditions=1,
            num_repetitions=1,
            max_concurrent=1,
        )
        observer.evaluation_completed(run_id="r", total_runs=2, elapsed_seconds=1.5)
        assert factory.bar.closed is True

    def test_bar_reference_cleared_after_completed(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=1,
            total_conditions=1,
            num_repetitions=1,
            max_concurrent=1,
        )
        observer.evaluation_completed(run_id="r", total_runs=1, elapsed_seconds=0.5)
        # Internal reference should be cleared.
        assert observer._bar is None

    def test_completed_before_started_does_not_raise(self) -> None:
        observer, _ = _make_observer()
        # _bar is None; should be a safe no-op.
        observer.evaluation_completed(run_id="r", total_runs=0, elapsed_seconds=0.0)


class TestProgressEvaluationObserverNoOps:
    """All other observer methods are no-ops — they do not touch the bar."""

    def test_sample_condition_started_does_not_update_bar(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=1,
            total_conditions=1,
            num_repetitions=1,
            max_concurrent=1,
        )
        observer.sample_condition_started(
            run_id="r", sample_idx="s0", condition="c", repetition_index=0
        )
        assert factory.bar.update_calls == []

    def test_sample_condition_completed_does_not_update_bar(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=1,
            total_conditions=1,
            num_repetitions=1,
            max_concurrent=1,
        )
        observer.sample_condition_completed(
            run_id="r", sample_idx="s0", condition="c", repetition_index=0
        )
        assert factory.bar.update_calls == []

    def test_sample_condition_failed_does_not_update_bar(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=1,
            total_conditions=1,
            num_repetitions=1,
            max_concurrent=1,
        )
        observer.sample_condition_failed(
            run_id="r",
            sample_idx="s0",
            condition="c",
            repetition_index=0,
            reason="bad",
        )
        assert factory.bar.update_calls == []

    def test_sample_condition_retry_does_not_update_bar(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=1,
            total_conditions=1,
            num_repetitions=1,
            max_concurrent=1,
        )
        observer.sample_condition_retry(
            run_id="r",
            sample_idx="s0",
            condition="c",
            repetition_index=0,
            attempt=1,
            reason="timeout",
            backoff_seconds=1.0,
        )
        assert factory.bar.update_calls == []
