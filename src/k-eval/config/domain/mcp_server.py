"""MCP server configuration models — discriminated union on `type` field."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class StdioMcpServer(BaseModel, frozen=True):
    """MCP server launched as a subprocess via stdio."""

    type: Literal["stdio"]
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class SseMcpServer(BaseModel, frozen=True):
    """MCP server reachable over Server-Sent Events."""

    type: Literal["sse"]
    url: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)


class HttpMcpServer(BaseModel, frozen=True):
    """MCP server reachable over HTTP."""

    type: Literal["http"]
    url: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)


# Discriminated union — Pydantic selects the correct subtype from the `type` field.
# The `type` statement (Python 3.13) works correctly with Pydantic's discriminated
# union: validated in test with model_validate and isinstance checks.
type McpServer = Annotated[
    StdioMcpServer | SseMcpServer | HttpMcpServer,
    Field(discriminator="type"),
]
