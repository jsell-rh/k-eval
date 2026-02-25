"""Agent configuration model."""

from pydantic import BaseModel, Field


class AgentConfig(BaseModel, frozen=True):
    type: str = Field(min_length=1)
    model: str = Field(min_length=1)
