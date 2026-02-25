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
    """Factory that produces a new SpyBar per call and records kwargs for each call."""

    def __init__(self) -> None:
        self.bars: list[SpyBar] = []
        self.all_kwargs: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SpyBar:
        bar = SpyBar()
        self.bars.append(bar)
        self.all_kwargs.append(kwargs)
        return bar


def _make_observer() -> tuple[ProgressEvaluationObserver, SpyTqdmFactory]:
    factory = SpyTqdmFactory()
    observer = ProgressEvaluationObserver(tqdm_factory=factory)
    return observer, factory


class TestProgressEvaluationObserverStarted:
    """evaluation_started creates tqdm bars: one per condition plus one overall."""

    def test_bar_created_on_evaluation_started(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=3,
            total_conditions=2,
            condition_names=["cond-a", "cond-b"],
            num_repetitions=1,
            max_concurrent=4,
        )
        # 2 condition bars + 1 overall bar
        assert len(factory.bars) == 3
        # Descs use real condition names (no ANSI in non-TTY)
        assert "cond-a" in factory.all_kwargs[0]["desc"]
        assert "cond-b" in factory.all_kwargs[1]["desc"]

    def test_bar_unit_is_triple(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=1,
            total_conditions=1,
            condition_names=["cond-a"],
            num_repetitions=1,
            max_concurrent=1,
        )
        assert factory.all_kwargs[0]["unit"] == "triple"

    def test_bar_not_created_before_evaluation_started(self) -> None:
        observer, factory = _make_observer()
        # No evaluation_started call — factory should never have been called.
        assert factory.bars == []

    def test_overall_bar_total_equals_all_triples(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=3,
            total_conditions=2,
            condition_names=["cond-a", "cond-b"],
            num_repetitions=2,
            max_concurrent=4,
        )
        # overall bar is the last one; total = 3 * 2 * 2 = 12
        assert factory.all_kwargs[-1]["total"] == 12

    def test_condition_bar_total_equals_samples_times_reps(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=3,
            total_conditions=2,
            condition_names=["cond-a", "cond-b"],
            num_repetitions=2,
            max_concurrent=4,
        )
        # per-condition total = 3 * 2 = 6
        assert factory.all_kwargs[0]["total"] == 6
        assert factory.all_kwargs[1]["total"] == 6


class TestProgressEvaluationObserverProgress:
    """evaluation_progress increments the matching condition bar and the overall bar."""

    def _start_one_condition(self, observer: ProgressEvaluationObserver) -> str:
        condition = "baseline"
        observer.evaluation_started(
            run_id="r",
            total_samples=5,
            total_conditions=1,
            condition_names=["baseline"],
            num_repetitions=1,
            max_concurrent=1,
        )
        return condition

    def test_single_progress_event_increments_condition_and_overall_bar(self) -> None:
        observer, factory = _make_observer()
        condition = self._start_one_condition(observer)
        observer.evaluation_progress(
            run_id="r", condition=condition, completed=1, total=5
        )
        # bars[0] is the condition bar, bars[1] is the overall bar
        assert factory.bars[0].update_calls == [1]
        assert factory.bars[1].update_calls == [1]

    def test_multiple_progress_events_each_increment_by_one(self) -> None:
        observer, factory = _make_observer()
        condition = self._start_one_condition(observer)
        for i in range(1, 4):
            observer.evaluation_progress(
                run_id="r", condition=condition, completed=i, total=5
            )
        assert factory.bars[0].update_calls == [1, 1, 1]
        assert factory.bars[1].update_calls == [1, 1, 1]

    def test_progress_before_started_does_not_raise(self) -> None:
        observer, factory = _make_observer()
        # No evaluation_started — bars list is empty; should be a safe no-op.
        observer.evaluation_progress(
            run_id="r", condition="baseline", completed=1, total=5
        )
        assert factory.bars == []

    def test_two_conditions_routed_to_separate_bars(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=2,
            total_conditions=2,
            condition_names=["alpha", "beta"],
            num_repetitions=1,
            max_concurrent=1,
        )
        # First progress event for "alpha" maps to bars[0].
        observer.evaluation_progress(
            run_id="r", condition="alpha", completed=1, total=4
        )
        # First progress event for "beta" maps to bars[1].
        observer.evaluation_progress(run_id="r", condition="beta", completed=2, total=4)
        # Second event for "alpha" again.
        observer.evaluation_progress(
            run_id="r", condition="alpha", completed=3, total=4
        )
        # bars[0] = alpha bar: 2 updates; bars[1] = beta bar: 1 update
        assert factory.bars[0].update_calls == [1, 1]
        assert factory.bars[1].update_calls == [1]
        # bars[2] = overall bar: 3 updates
        assert factory.bars[2].update_calls == [1, 1, 1]


class TestProgressEvaluationObserverCompleted:
    """evaluation_completed closes all bars."""

    def test_all_bars_closed_on_evaluation_completed(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=2,
            total_conditions=2,
            condition_names=["cond-a", "cond-b"],
            num_repetitions=1,
            max_concurrent=1,
        )
        observer.evaluation_completed(run_id="r", total_runs=4, elapsed_seconds=1.5)
        assert all(bar.closed for bar in factory.bars)

    def test_overall_bar_reference_cleared_after_completed(self) -> None:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=1,
            total_conditions=1,
            condition_names=["cond-a"],
            num_repetitions=1,
            max_concurrent=1,
        )
        observer.evaluation_completed(run_id="r", total_runs=1, elapsed_seconds=0.5)
        assert observer._overall_bar is None

    def test_completed_before_started_does_not_raise(self) -> None:
        observer, _ = _make_observer()
        # _overall_bar is None and _condition_bars is empty; should be a safe no-op.
        observer.evaluation_completed(run_id="r", total_runs=0, elapsed_seconds=0.0)


class TestProgressEvaluationObserverNoOps:
    """All other observer methods are no-ops — they do not touch the bars."""

    def _started_observer(self) -> tuple[ProgressEvaluationObserver, SpyTqdmFactory]:
        observer, factory = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=1,
            total_conditions=1,
            condition_names=["cond-a"],
            num_repetitions=1,
            max_concurrent=1,
        )
        return observer, factory

    def test_sample_condition_started_does_not_update_bar(self) -> None:
        observer, factory = self._started_observer()
        observer.sample_condition_started(
            run_id="r", sample_idx="s0", condition="c", repetition_index=0
        )
        assert all(bar.update_calls == [] for bar in factory.bars)

    def test_sample_condition_completed_does_not_update_bar(self) -> None:
        observer, factory = self._started_observer()
        observer.sample_condition_completed(
            run_id="r", sample_idx="s0", condition="c", repetition_index=0
        )
        assert all(bar.update_calls == [] for bar in factory.bars)

    def test_sample_condition_failed_does_not_update_bar(self) -> None:
        observer, factory = self._started_observer()
        observer.sample_condition_failed(
            run_id="r",
            sample_idx="s0",
            condition="c",
            repetition_index=0,
            reason="bad",
        )
        assert all(bar.update_calls == [] for bar in factory.bars)

    def test_sample_condition_retry_does_not_update_bar(self) -> None:
        observer, factory = self._started_observer()
        observer.sample_condition_retry(
            run_id="r",
            sample_idx="s0",
            condition="c",
            repetition_index=0,
            attempt=1,
            reason="timeout",
            backoff_seconds=1.0,
        )
        assert all(bar.update_calls == [] for bar in factory.bars)
