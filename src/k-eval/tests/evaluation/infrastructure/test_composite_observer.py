"""Tests for CompositeEvaluationObserver."""

from k_eval.evaluation.infrastructure.composite_observer import (
    CompositeEvaluationObserver,
)
from tests.evaluation.fake_observer import FakeEvaluationObserver


def _make_composite(
    *observers: FakeEvaluationObserver,
) -> CompositeEvaluationObserver:
    return CompositeEvaluationObserver(observers=list(observers))


class TestCompositeEvaluationObserverFanOut:
    """Every event is forwarded to all observers in order."""

    def test_evaluation_started_forwarded_to_all(self) -> None:
        obs_a = FakeEvaluationObserver()
        obs_b = FakeEvaluationObserver()
        composite = _make_composite(obs_a, obs_b)

        composite.evaluation_started(
            run_id="run-1",
            total_samples=3,
            total_conditions=2,
            condition_names=["baseline", "other"],
            num_repetitions=1,
            max_concurrent=4,
        )

        assert len(obs_a.started) == 1
        assert len(obs_b.started) == 1
        assert obs_a.started[0].run_id == "run-1"
        assert obs_b.started[0].run_id == "run-1"

    def test_evaluation_started_preserves_all_fields(self) -> None:
        obs = FakeEvaluationObserver()
        composite = _make_composite(obs)

        composite.evaluation_started(
            run_id="run-abc",
            total_samples=5,
            total_conditions=3,
            condition_names=["baseline", "other", "third"],
            num_repetitions=2,
            max_concurrent=8,
        )

        event = obs.started[0]
        assert event.run_id == "run-abc"
        assert event.total_samples == 5
        assert event.total_conditions == 3
        assert event.num_repetitions == 2
        assert event.max_concurrent == 8

    def test_evaluation_completed_forwarded_to_all(self) -> None:
        obs_a = FakeEvaluationObserver()
        obs_b = FakeEvaluationObserver()
        composite = _make_composite(obs_a, obs_b)

        composite.evaluation_completed(
            run_id="run-1",
            total_runs=6,
            elapsed_seconds=12.5,
        )

        assert len(obs_a.completed) == 1
        assert len(obs_b.completed) == 1
        assert obs_a.completed[0].elapsed_seconds == 12.5
        assert obs_b.completed[0].elapsed_seconds == 12.5

    def test_evaluation_completed_preserves_all_fields(self) -> None:
        obs = FakeEvaluationObserver()
        composite = _make_composite(obs)

        composite.evaluation_completed(
            run_id="run-xyz",
            total_runs=9,
            elapsed_seconds=3.7,
        )

        event = obs.completed[0]
        assert event.run_id == "run-xyz"
        assert event.total_runs == 9
        assert event.elapsed_seconds == 3.7

    def test_evaluation_progress_forwarded_to_all(self) -> None:
        obs_a = FakeEvaluationObserver()
        obs_b = FakeEvaluationObserver()
        composite = _make_composite(obs_a, obs_b)

        composite.evaluation_progress(
            run_id="run-1", condition="baseline", completed=3, total=10
        )

        assert len(obs_a.progress) == 1
        assert len(obs_b.progress) == 1
        assert obs_a.progress[0].completed == 3
        assert obs_b.progress[0].completed == 3

    def test_evaluation_progress_preserves_all_fields(self) -> None:
        obs = FakeEvaluationObserver()
        composite = _make_composite(obs)

        composite.evaluation_progress(
            run_id="run-abc", condition="baseline", completed=7, total=20
        )

        event = obs.progress[0]
        assert event.run_id == "run-abc"
        assert event.condition == "baseline"
        assert event.completed == 7
        assert event.total == 20

    def test_sample_condition_started_forwarded_to_all(self) -> None:
        obs_a = FakeEvaluationObserver()
        obs_b = FakeEvaluationObserver()
        composite = _make_composite(obs_a, obs_b)

        composite.sample_condition_started(
            run_id="run-1",
            sample_idx="s0",
            condition="baseline",
            repetition_index=0,
        )

        assert len(obs_a.sc_started) == 1
        assert len(obs_b.sc_started) == 1

    def test_sample_condition_completed_forwarded_to_all(self) -> None:
        obs_a = FakeEvaluationObserver()
        obs_b = FakeEvaluationObserver()
        composite = _make_composite(obs_a, obs_b)

        composite.sample_condition_completed(
            run_id="run-1",
            sample_idx="s0",
            condition="baseline",
            repetition_index=0,
        )

        assert len(obs_a.sc_completed) == 1
        assert len(obs_b.sc_completed) == 1

    def test_sample_condition_failed_forwarded_to_all(self) -> None:
        obs_a = FakeEvaluationObserver()
        obs_b = FakeEvaluationObserver()
        composite = _make_composite(obs_a, obs_b)

        composite.sample_condition_failed(
            run_id="run-1",
            sample_idx="s0",
            condition="baseline",
            repetition_index=0,
            reason="agent error",
        )

        assert len(obs_a.sc_failed) == 1
        assert len(obs_b.sc_failed) == 1
        assert obs_a.sc_failed[0].reason == "agent error"
        assert obs_b.sc_failed[0].reason == "agent error"

    def test_sample_condition_retry_forwarded_to_all(self) -> None:
        obs_a = FakeEvaluationObserver()
        obs_b = FakeEvaluationObserver()
        composite = _make_composite(obs_a, obs_b)

        composite.sample_condition_retry(
            run_id="run-1",
            sample_idx="s0",
            condition="baseline",
            repetition_index=0,
            attempt=2,
            reason="timeout",
            backoff_seconds=1.5,
        )

        assert len(obs_a.sc_retried) == 1
        assert len(obs_b.sc_retried) == 1
        assert obs_a.sc_retried[0].backoff_seconds == 1.5
        assert obs_b.sc_retried[0].backoff_seconds == 1.5


class TestCompositeEvaluationObserverOrder:
    """Events are delivered to observers in the order they were registered."""

    def test_started_events_delivered_in_registration_order(self) -> None:
        received_order: list[str] = []

        class OrderRecorder:
            def __init__(self, name: str) -> None:
                self._name = name

            def evaluation_started(
                self,
                run_id: str,
                total_samples: int,
                total_conditions: int,
                condition_names: list[str],
                num_repetitions: int,
                max_concurrent: int,
            ) -> None:  # noqa: E501
                received_order.append(self._name)

            def evaluation_completed(
                self, run_id: str, total_runs: int, elapsed_seconds: float
            ) -> None:
                pass

            def evaluation_progress(
                self,
                run_id: str,
                condition: str,
                completed: int,
                total: int,
            ) -> None:
                pass

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

        first = OrderRecorder(name="first")
        second = OrderRecorder(name="second")
        third = OrderRecorder(name="third")
        composite = CompositeEvaluationObserver(observers=[first, second, third])

        composite.evaluation_started(
            run_id="r",
            total_samples=1,
            total_conditions=1,
            condition_names=["baseline"],
            num_repetitions=1,
            max_concurrent=1,
        )

        assert received_order == ["first", "second", "third"]


class TestCompositeEvaluationObserverEmpty:
    """An empty observer list is valid and events are silently dropped."""

    def test_evaluation_started_with_no_observers(self) -> None:
        composite = CompositeEvaluationObserver(observers=[])
        # Should not raise.
        composite.evaluation_started(
            run_id="r",
            total_samples=1,
            total_conditions=1,
            condition_names=["baseline"],
            num_repetitions=1,
            max_concurrent=1,
        )

    def test_evaluation_progress_with_no_observers(self) -> None:
        composite = CompositeEvaluationObserver(observers=[])
        composite.evaluation_progress(
            run_id="r", condition="baseline", completed=1, total=5
        )

    def test_evaluation_completed_with_no_observers(self) -> None:
        composite = CompositeEvaluationObserver(observers=[])
        composite.evaluation_completed(run_id="r", total_runs=5, elapsed_seconds=1.0)


class TestCompositeEvaluationObserverSingleObserver:
    """Smoke-test that a single-observer composite behaves identically to the observer."""

    def test_single_observer_receives_all_event_types(self) -> None:
        obs = FakeEvaluationObserver()
        composite = _make_composite(obs)

        composite.evaluation_started(
            run_id="r",
            total_samples=2,
            total_conditions=1,
            condition_names=["c"],
            num_repetitions=1,
            max_concurrent=2,
        )
        composite.evaluation_progress(run_id="r", condition="c", completed=1, total=2)
        composite.sample_condition_started(
            run_id="r", sample_idx="s0", condition="c", repetition_index=0
        )
        composite.sample_condition_completed(
            run_id="r", sample_idx="s0", condition="c", repetition_index=0
        )
        composite.sample_condition_retry(
            run_id="r",
            sample_idx="s1",
            condition="c",
            repetition_index=0,
            attempt=1,
            reason="err",
            backoff_seconds=0.5,
        )
        composite.sample_condition_failed(
            run_id="r",
            sample_idx="s1",
            condition="c",
            repetition_index=0,
            reason="final err",
        )
        composite.evaluation_progress(run_id="r", condition="c", completed=2, total=2)
        composite.evaluation_completed(run_id="r", total_runs=2, elapsed_seconds=0.1)

        assert len(obs.started) == 1
        assert len(obs.progress) == 2
        assert len(obs.sc_started) == 1
        assert len(obs.sc_completed) == 1
        assert len(obs.sc_retried) == 1
        assert len(obs.sc_failed) == 1
        assert len(obs.completed) == 1
