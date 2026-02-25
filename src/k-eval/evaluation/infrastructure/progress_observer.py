"""ProgressEvaluationObserver — writes per-condition and overall tqdm bars to stderr."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, Protocol, cast

from tqdm import tqdm

# ANSI color cycle for condition bars (reset with \033[0m)
_CONDITION_COLORS: list[str] = [
    "\033[36m",  # cyan
    "\033[32m",  # green
    "\033[33m",  # yellow
    "\033[35m",  # magenta
    "\033[34m",  # blue
]
_RESET = "\033[0m"


class ProgressBar(Protocol):
    """Structural interface for the subset of tqdm's API that we use."""

    def update(self, n: int = ...) -> bool | None: ...
    def close(self) -> None: ...


type TqdmFactory = Callable[..., ProgressBar]


def _default_tqdm_factory(**kwargs: Any) -> ProgressBar:
    return cast(ProgressBar, tqdm(**kwargs))


class ProgressEvaluationObserver:
    """Renders per-condition tqdm bars plus an overall bar on stderr.

    One bar is created per condition (at positions 0, 1, 2, …) using the real
    condition names passed to evaluation_started, and an overall bar is placed
    at position len(condition_names). All descs are padded to the same width so
    the bars line up. ANSI colour is applied to condition labels when stderr is
    a TTY.

    Only evaluation_started, evaluation_progress, and evaluation_completed
    produce output; all other events are no-ops.

    The ``tqdm_factory`` parameter is intended for testing — pass a factory
    that returns any object satisfying the ProgressBar protocol (update/close).

    Does NOT inherit from EvaluationObserver (structural typing via Protocol).
    """

    def __init__(self, tqdm_factory: TqdmFactory = _default_tqdm_factory) -> None:
        self._tqdm_factory = tqdm_factory
        self._condition_bars: dict[str, ProgressBar] = {}
        self._overall_bar: ProgressBar | None = None

    def evaluation_started(
        self,
        run_id: str,
        total_samples: int,
        total_conditions: int,
        condition_names: list[str],
        num_repetitions: int,
        max_concurrent: int,
    ) -> None:
        # Clear any leftover state from a previous run.
        self._condition_bars = {}
        self._overall_bar = None

        use_color = sys.stderr.isatty()
        per_condition_total = total_samples * num_repetitions
        overall_total = total_samples * total_conditions * num_repetitions

        # Compute the desc width so all bars are aligned.
        desc_width = max(
            (len(name) for name in condition_names + ["Overall"]),
            default=len("Overall"),
        )

        # Create one bar per condition using the real condition name.
        for i, name in enumerate(condition_names):
            color = _CONDITION_COLORS[i % len(_CONDITION_COLORS)] if use_color else ""
            reset = _RESET if use_color else ""
            bar = self._tqdm_factory(
                total=per_condition_total,
                desc=f"{color}{name:<{desc_width}}{reset}",
                unit="triple",
                file=sys.stderr,
                dynamic_ncols=True,
                position=i,
                leave=True,
            )
            self._condition_bars[name] = bar

        # Create the overall bar after all condition bars.
        self._overall_bar = self._tqdm_factory(
            total=overall_total,
            desc=f"{'Overall':<{desc_width}}",
            unit="triple",
            file=sys.stderr,
            dynamic_ncols=True,
            position=len(condition_names),
            leave=True,
        )

    def evaluation_completed(
        self,
        run_id: str,
        total_runs: int,
        elapsed_seconds: float,
    ) -> None:
        for bar in self._condition_bars.values():
            bar.close()
        if self._overall_bar is not None:
            self._overall_bar.close()
        self._condition_bars = {}
        self._overall_bar = None

    def evaluation_progress(
        self,
        run_id: str,
        condition: str,
        completed: int,
        total: int,
    ) -> None:
        if not self._condition_bars and self._overall_bar is None:
            return

        if condition in self._condition_bars:
            self._condition_bars[condition].update(n=1)

        if self._overall_bar is not None:
            self._overall_bar.update(n=1)

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
