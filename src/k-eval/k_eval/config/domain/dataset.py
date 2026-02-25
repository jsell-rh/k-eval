"""Dataset configuration model."""

from pathlib import Path

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel, frozen=True):
    path: Path
    question_key: str = Field(min_length=1)
    answer_key: str = Field(min_length=1)
