"""ProgressEvaluationObserver — renders per-condition Rich progress bars to stderr."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

# ANSI color names for condition descriptions (Rich markup style).
_CONDITION_COLORS: list[str] = [
    "cyan",
    "green",
    "yellow",
    "magenta",
    "blue",
]


class _ThreeSegmentBarColumn(BarColumn):
    """BarColumn subclass that renders three segments: done, in-flight, remaining."""

    def render(self, task: Any) -> Text:  # type: ignore[override]
        bar_width = self.bar_width or 40
        total = task.total or 0
        if total > 0:
            done_cells = round(task.completed / total * bar_width)
            inflight = task.fields.get("inflight", 0)
            inflight_cells = min(
                round(inflight / total * bar_width),
                bar_width - done_cells,
            )
        else:
            done_cells = 0
            inflight_cells = 0
        remaining_cells = bar_width - done_cells - inflight_cells

        result = Text()
        result.append("█" * done_cells, style="bright_green")
        result.append("█" * inflight_cells, style="yellow")
        result.append("░" * remaining_cells, style="dim white")
        return result


class ProgressEvaluationObserver:
    """Renders per-condition Rich progress bars plus an overall bar on stderr.

    One row is created per condition plus one Overall row. ANSI colour is applied
    to condition labels when stderr is a TTY.

    Only evaluation_started, evaluation_progress, sample_condition_started, and
    evaluation_completed produce output; all other events are no-ops.

    Pass ``disabled=True`` to suppress all terminal output (useful in tests).

    Does NOT inherit from EvaluationObserver (structural typing via Protocol).
    """

    def __init__(self, disabled: bool = False) -> None:
        self._disabled = disabled
        self._done: dict[str, int] = {}
        self._inflight: dict[str, int] = {}
        self._total: dict[str, int] = {}
        self._task_ids: dict[str, TaskID] = {}
        self._progress: Progress | None = None
        self._live: Live | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_desc(self, name: str, index: int, pad_width: int) -> str:
        """Build a description string for a condition row, with optional color."""
        if name == "Overall":
            return f"{'Overall':<{pad_width}}"
        use_color = sys.stderr.isatty()
        if use_color:
            color = _CONDITION_COLORS[index % len(_CONDITION_COLORS)]
            return f"[{color}]{name:<{pad_width}}[/{color}]"
        return f"{name:<{pad_width}}"

    def _rate_str(self, key: str) -> str:
        """Compute a rate string like '1.2 triple/s' or '-- triple/s'."""
        if self._progress is None:
            return "-- triple/s"
        task_id = self._task_ids.get(key)
        if task_id is None:
            return "-- triple/s"
        task = self._progress.tasks[task_id]
        elapsed = task.elapsed
        if elapsed is not None and elapsed > 0 and task.completed > 0:
            rate = task.completed / elapsed
            return f"{rate:.1f} triple/s"
        return "-- triple/s"

    def _update_task(self, key: str) -> None:
        """Push current _done/_inflight state into the Rich task."""
        if self._progress is None or key not in self._task_ids:
            return
        done = self._done.get(key, 0)
        inflight = self._inflight.get(key, 0)
        rate = self._rate_str(key=key)
        self._progress.update(
            self._task_ids[key],
            completed=done,
            inflight=inflight,
            done=done,
            rate=rate,
        )

    # ------------------------------------------------------------------
    # Observer events
    # ------------------------------------------------------------------

    def evaluation_started(
        self,
        run_id: str,
        total_samples: int,
        total_conditions: int,
        condition_names: list[str],
        num_repetitions: int,
        max_concurrent: int,
    ) -> None:
        # Reset state from any previous run.
        self._done = {}
        self._inflight = {}
        self._total = {}
        self._task_ids = {}
        self._progress = None
        self._live = None

        per_condition_total = total_samples * num_repetitions
        overall_total = total_samples * total_conditions * num_repetitions

        # Populate totals.
        for name in condition_names:
            self._total[name] = per_condition_total
            self._done[name] = 0
            self._inflight[name] = 0
        self._total["Overall"] = overall_total
        self._done["Overall"] = 0
        self._inflight["Overall"] = 0

        if self._disabled:
            return

        # Print legend once before starting the live display.
        print(
            "  Legend:  \033[92m█\033[0m done"
            "  \033[33m█\033[0m in-flight"
            "  \033[2m░\033[0m remaining",
            file=sys.stderr,
        )

        pad_width = max(
            (len(name) for name in condition_names + ["Overall"]),
            default=len("Overall"),
        )

        console = Console(stderr=True)
        self._progress = Progress(
            TextColumn("{task.description}"),
            _ThreeSegmentBarColumn(bar_width=None),
            TextColumn("{task.fields[done]}+{task.fields[inflight]}/{task.total:.0f}"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[rate]}"),
            console=console,
            refresh_per_second=10,
            transient=False,
        )

        # Create one task per condition, then the Overall task.
        for i, name in enumerate(condition_names):
            desc = self._make_desc(name=name, index=i, pad_width=pad_width)
            task_id = self._progress.add_task(
                description=desc,
                total=float(per_condition_total),
                inflight=0,
                done=0,
                rate="-- triple/s",
            )
            self._task_ids[name] = task_id

        overall_desc = self._make_desc(name="Overall", index=0, pad_width=pad_width)
        overall_task_id = self._progress.add_task(
            description=overall_desc,
            total=float(overall_total),
            inflight=0,
            done=0,
            rate="-- triple/s",
        )
        self._task_ids["Overall"] = overall_task_id

        self._live = Live(self._progress, console=console, refresh_per_second=10)
        self._live.start()

    def evaluation_completed(
        self,
        run_id: str,
        total_runs: int,
        elapsed_seconds: float,
    ) -> None:
        if self._live is not None:
            self._live.stop()

        self._done = {}
        self._inflight = {}
        self._total = {}
        self._task_ids = {}
        self._progress = None
        self._live = None

    def evaluation_progress(
        self,
        run_id: str,
        condition: str,
        completed: int,
        total: int,
    ) -> None:
        if condition in self._done:
            self._done[condition] += 1
            self._inflight[condition] = max(0, self._inflight[condition] - 1)
        if "Overall" in self._done:
            self._done["Overall"] += 1
            self._inflight["Overall"] = max(0, self._inflight["Overall"] - 1)

        if not self._disabled:
            self._update_task(key=condition)
            self._update_task(key="Overall")

    def sample_condition_started(
        self,
        run_id: str,
        sample_idx: str,
        condition: str,
        repetition_index: int,
    ) -> None:
        if condition in self._inflight:
            self._inflight[condition] += 1
        if "Overall" in self._inflight:
            self._inflight["Overall"] += 1

        if not self._disabled:
            self._update_task(key=condition)
            self._update_task(key="Overall")

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
