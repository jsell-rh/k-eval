"""Error types raised by judge infrastructure."""

from k_eval.core.errors import KEvalError


class JudgeInvocationError(KEvalError):
    """Raised when the judge cannot be invoked or returns an unparseable response."""

    def __init__(self, reason: str, retriable: bool = False) -> None:
        super().__init__(f"Failed to score response: {reason}", retriable=retriable)
