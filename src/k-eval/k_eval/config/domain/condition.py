"""Evaluation condition configuration model."""

from pydantic import BaseModel, Field

from k_eval.config.domain.condition_mcp_server import ConditionMcpServer


class ConditionConfig(BaseModel, frozen=True):
    """A single evaluation condition — a named configuration variant."""

    mcp_servers: list[ConditionMcpServer]
    system_prompt: str = Field(min_length=1)
    require_mcp_tool_use: bool = False
    require_mcp_tool_success: bool = False
