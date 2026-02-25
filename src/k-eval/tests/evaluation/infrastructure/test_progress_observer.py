"""Tests for ProgressEvaluationObserver."""

import pytest

from evaluation.infrastructure.progress_observer import ProgressEvaluationObserver


def _make_observer() -> ProgressEvaluationObserver:
    return ProgressEvaluationObserver()


class TestProgressEvaluationObserverProgressBar:
    """evaluation_progress writes a progress bar to stderr."""

    def test_progress_writes_to_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_progress(run_id="r", completed=1, total=10)
        captured = capsys.readouterr()
        assert len(captured.err) > 0

    def test_progress_contains_completed_and_total(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_progress(run_id="r", completed=3, total=10)
        captured = capsys.readouterr()
        assert "3/10" in captured.err

    def test_progress_contains_percentage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_progress(run_id="r", completed=5, total=10)
        captured = capsys.readouterr()
        assert "50%" in captured.err

    def test_progress_uses_carriage_return(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_progress(run_id="r", completed=1, total=5)
        captured = capsys.readouterr()
        assert "\r" in captured.err

    def test_progress_uses_erase_line_escape(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_progress(run_id="r", completed=1, total=5)
        captured = capsys.readouterr()
        assert "\033[K" in captured.err

    def test_progress_bar_contains_filled_blocks(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_progress(run_id="r", completed=10, total=10)
        captured = capsys.readouterr()
        assert "█" in captured.err

    def test_progress_bar_contains_empty_blocks_when_partial(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_progress(run_id="r", completed=0, total=10)
        captured = capsys.readouterr()
        assert "░" in captured.err

    def test_progress_at_zero_of_total_shows_zero_percent(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_progress(run_id="r", completed=0, total=10)
        captured = capsys.readouterr()
        assert "0%" in captured.err

    def test_progress_at_100_percent_shows_100(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_progress(run_id="r", completed=10, total=10)
        captured = capsys.readouterr()
        assert "100%" in captured.err

    def test_progress_zero_total_does_not_raise(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        # Should not raise ZeroDivisionError.
        observer.evaluation_progress(run_id="r", completed=0, total=0)

    def test_progress_nothing_written_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_progress(run_id="r", completed=2, total=5)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestProgressEvaluationObserverCompleted:
    """evaluation_completed overwrites the bar with a summary line."""

    def test_completed_writes_to_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_completed(run_id="r", total_runs=6, elapsed_seconds=5.0)
        captured = capsys.readouterr()
        assert len(captured.err) > 0

    def test_completed_contains_total_runs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_completed(run_id="r", total_runs=9, elapsed_seconds=3.0)
        captured = capsys.readouterr()
        assert "9" in captured.err

    def test_completed_ends_with_newline(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_completed(run_id="r", total_runs=4, elapsed_seconds=2.0)
        captured = capsys.readouterr()
        assert captured.err.endswith("\n")

    def test_completed_uses_seconds_only_for_short_runs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_completed(run_id="r", total_runs=1, elapsed_seconds=45.3)
        captured = capsys.readouterr()
        assert "45.3s" in captured.err
        # Minutes format looks like "1m 30.0s" — no digit-followed-by-m pattern for sub-minute runs.
        import re

        assert not re.search(r"\d+m", captured.err)

    def test_completed_uses_minutes_and_seconds_for_long_runs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        # 90 seconds = 1m 30.0s
        observer.evaluation_completed(run_id="r", total_runs=1, elapsed_seconds=90.0)
        captured = capsys.readouterr()
        assert "1m" in captured.err
        assert "30.0s" in captured.err

    def test_completed_uses_carriage_return(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_completed(run_id="r", total_runs=2, elapsed_seconds=1.0)
        captured = capsys.readouterr()
        assert "\r" in captured.err

    def test_completed_nothing_written_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_completed(run_id="r", total_runs=2, elapsed_seconds=1.0)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestProgressEvaluationObserverNoOps:
    """All other observer methods are no-ops — they produce no output."""

    def test_evaluation_started_is_noop(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.evaluation_started(
            run_id="r",
            total_samples=2,
            total_conditions=1,
            num_repetitions=1,
            max_concurrent=4,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_sample_condition_started_is_noop(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.sample_condition_started(
            run_id="r", sample_idx="s0", condition="c", repetition_index=0
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_sample_condition_completed_is_noop(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.sample_condition_completed(
            run_id="r", sample_idx="s0", condition="c", repetition_index=0
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_sample_condition_failed_is_noop(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.sample_condition_failed(
            run_id="r",
            sample_idx="s0",
            condition="c",
            repetition_index=0,
            reason="bad",
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_sample_condition_retry_is_noop(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        observer.sample_condition_retry(
            run_id="r",
            sample_idx="s0",
            condition="c",
            repetition_index=0,
            attempt=1,
            reason="timeout",
            backoff_seconds=1.0,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestProgressEvaluationObserverBarWidth:
    """The bar always has the expected fixed width of filled + empty blocks."""

    def test_bar_total_width_is_constant(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observer = _make_observer()
        bar_width = observer._BAR_WIDTH

        # Test at several completion levels.
        for completed in [0, 3, 7, 10]:
            observer.evaluation_progress(run_id="r", completed=completed, total=10)
            captured = capsys.readouterr()
            filled = captured.err.count("█")
            empty = captured.err.count("░")
            assert filled + empty == bar_width, (
                f"completed={completed}: expected bar width {bar_width}, got {filled + empty}"
            )
