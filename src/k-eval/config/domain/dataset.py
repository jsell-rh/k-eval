"""Dataset configuration model."""

from pathlib import Path

from pydantic import BaseModel


class DatasetConfig(BaseModel, frozen=True):
    path: Path
    question_key: str
    answer_key: str
