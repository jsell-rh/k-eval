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
