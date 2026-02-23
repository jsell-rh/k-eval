"""Error types raised by agent infrastructure."""

from core.errors import KEvalError


class AgentInvocationError(KEvalError):
    """Raised when the agent cannot be invoked or returns an error response."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Failed to invoke agent: {reason}")
