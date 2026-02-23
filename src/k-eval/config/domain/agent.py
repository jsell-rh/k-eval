"""Agent configuration model."""

from pydantic import BaseModel


class AgentConfig(BaseModel, frozen=True):
    type: str
    model: str
