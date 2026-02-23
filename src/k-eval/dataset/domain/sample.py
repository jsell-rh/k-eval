"""Sample domain value object — one question/answer pair from a dataset."""

from pydantic import BaseModel, ConfigDict


class Sample(BaseModel, frozen=True):
    """Immutable value object representing a single question/answer pair."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    answer: str
