"""UsageMetrics value object — token usage from an agent invocation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UsageMetrics:
    """Immutable value object capturing token usage from a single agent invocation."""

    input_tokens: int | None
    output_tokens: int | None
