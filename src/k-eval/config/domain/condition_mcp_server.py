"""ConditionMcpServer — a resolved MCP server reference within an evaluation condition."""

from pydantic.dataclasses import dataclass

from config.domain.mcp_server import McpServer


@dataclass(frozen=True)
class ConditionMcpServer:
    """A resolved MCP server reference: the server name paired with its full config."""

    name: str
    config: McpServer
