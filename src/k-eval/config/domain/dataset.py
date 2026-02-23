"""Dataset configuration model."""

from pydantic import BaseModel


class DatasetConfig(BaseModel, frozen=True):
    path: str
    question_key: str
    answer_key: str
