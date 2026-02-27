"""Error types raised by agent infrastructure."""

from k_eval.core.errors import KEvalError


class AgentInvocationError(KEvalError):
    """Raised when the agent cannot be invoked or returns an error response."""

    def __init__(self, reason: str, retriable: bool = False) -> None:
        super().__init__(f"Failed to invoke agent: {reason}", retriable=retriable)


class AgentTypeNotSupportedError(KEvalError):
    """Raised when the agent type specified in config is not a known agent type."""

    def __init__(self, agent_type: str) -> None:
        super().__init__(
            f"Failed to create agent factory: unsupported agent type '{agent_type}'"
        )


class McpToolUseAbsentError(KEvalError):
    """Raised when a condition requires MCP tool use but the agent made no tool calls."""

    def __init__(self, condition: str, sample_idx: int) -> None:
        super().__init__(
            f"Failed to verify MCP tool use: condition '{condition}' requires at least"
            f" one MCP tool call but agent called none",
            retriable=True,
        )


class McpToolSuccessAbsentError(KEvalError):
    """Raised when all MCP tool calls in a session resulted in errors."""

    def __init__(self, condition: str, sample_idx: int) -> None:
        super().__init__(
            f"Failed to verify MCP tool success: condition '{condition}' requires at"
            f" least one successful MCP tool call but all calls errored",
            retriable=True,
        )
