"""JudgeFactory Protocol — structural interface for constructing Judge instances."""

from typing import Protocol

from judge.domain.judge import Judge


class JudgeFactory(Protocol):
    """Constructs a new Judge instance for a given (condition, sample) pair."""

    def create(
        self,
        condition: str,
        sample_idx: str,
    ) -> Judge: ...
