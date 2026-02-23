"""Judge configuration model."""

from pydantic import BaseModel, Field


class JudgeConfig(BaseModel, frozen=True):
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0)
