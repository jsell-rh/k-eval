"""MCP server configuration models — discriminated union on `type` field."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class StdioMcpServer(BaseModel, frozen=True):
    """MCP server launched as a subprocess via stdio."""

    type: Literal["stdio"]
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class SseMcpServer(BaseModel, frozen=True):
    """MCP server reachable over Server-Sent Events."""

    type: Literal["sse"]
    url: str
    headers: dict[str, str] | None = None


class HttpMcpServer(BaseModel, frozen=True):
    """MCP server reachable over HTTP."""

    type: Literal["http"]
    url: str
    headers: dict[str, str] | None = None


# Discriminated union — Pydantic selects the correct subtype from the `type` field.
McpServer = Annotated[
    StdioMcpServer | SseMcpServer | HttpMcpServer,
    Field(discriminator="type"),
]
