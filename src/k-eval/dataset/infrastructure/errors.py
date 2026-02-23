"""Error types raised by dataset infrastructure."""

from core.errors import KEvalError


class DatasetLoadError(KEvalError):
    """Raised when a JSONL dataset cannot be loaded or is malformed."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Failed to load dataset: {reason}")
