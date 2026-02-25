"""UsageMetrics value object — token usage from an agent invocation."""

from pydantic import BaseModel, ConfigDict


class UsageMetrics(BaseModel, frozen=True):
    """Immutable value object capturing token usage from a single agent invocation."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int | None
    output_tokens: int | None
