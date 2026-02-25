"""ProgressEvaluationObserver — writes a live tqdm progress bar to stderr."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, cast

from tqdm import tqdm


type TqdmFactory = Callable[..., Any]


def _default_tqdm_factory(**kwargs: Any) -> tqdm[int]:
    return cast(tqdm[int], tqdm(**kwargs))


class ProgressEvaluationObserver:
    """Renders a live tqdm progress bar on stderr during an evaluation run.

    Only evaluation_started, evaluation_progress, and evaluation_completed
    produce output; all other events are no-ops.

    The ``tqdm_factory`` parameter is intended for testing — pass a factory
    that returns a tqdm instance configured with ``disable=True`` or a spy
    to assert on calls without terminal output.

    Does NOT inherit from EvaluationObserver (structural typing via Protocol).
    """

    def __init__(self, tqdm_factory: TqdmFactory = _default_tqdm_factory) -> None:
        self._tqdm_factory = tqdm_factory
        self._bar: tqdm[int] | None = None

    def evaluation_started(
        self,
        run_id: str,
        total_samples: int,
        total_conditions: int,
        num_repetitions: int,
        max_concurrent: int,
    ) -> None:
        total = total_samples * total_conditions * num_repetitions
        self._bar = self._tqdm_factory(
            total=total,
            desc="Evaluating",
            unit="triple",
            file=sys.stderr,
            dynamic_ncols=True,
        )

    def evaluation_completed(
        self,
        run_id: str,
        total_runs: int,
        elapsed_seconds: float,
    ) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def evaluation_progress(
        self,
        run_id: str,
        completed: int,
        total: int,
    ) -> None:
        if self._bar is not None:
            self._bar.update(n=1)

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
