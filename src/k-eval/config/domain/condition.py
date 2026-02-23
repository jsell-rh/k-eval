"""Evaluation condition configuration model."""

from pydantic import BaseModel


class ConditionConfig(BaseModel, frozen=True):
    """A single evaluation condition — a named configuration variant."""

    mcp_servers: list[str]
    system_prompt: str
