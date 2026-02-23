"""Judge configuration model."""

from pydantic import BaseModel


class JudgeConfig(BaseModel, frozen=True):
    model: str
    temperature: float
