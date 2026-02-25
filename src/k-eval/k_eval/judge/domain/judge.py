"""Judge Protocol — structural interface for all judge implementations."""

from typing import Protocol

from k_eval.judge.domain.score import JudgeResult


class Judge(Protocol):
    """Structural interface satisfied by any judge implementation.

    Each instance is constructed once per (condition, sample) evaluation.
    """

    async def score(
        self, question: str, golden_answer: str, agent_response: str
    ) -> JudgeResult: ...
