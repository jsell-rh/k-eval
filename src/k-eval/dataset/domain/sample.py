"""Sample domain value object — one question/answer pair from a dataset."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    """Immutable value object representing a single question/answer pair."""

    id: str
    question: str
    answer: str
