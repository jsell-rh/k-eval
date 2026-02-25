"""Base exception class for all k-eval-specific errors."""


class KEvalError(Exception):
    """Base class for all k-eval errors."""

    def __init__(self, message: str, retriable: bool = False) -> None:
        super().__init__(message)
        self.retriable = retriable
