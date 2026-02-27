"""ToolCall and AgentTurn domain value objects — structured turn capture from agent streams."""

from typing import Literal

from pydantic import BaseModel


class ToolCall(BaseModel, frozen=True):
    """One MCP tool invocation and its result, captured from the agent stream."""

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, object]
    tool_result: str | None  # None if error or no response received
    tool_error: bool  # True if ToolResultBlock.is_error was truthy
    duration_ms: float | None = None  # Wall-clock duration; None for unresolved calls


class AgentTurn(BaseModel, frozen=True):
    """One conversational turn captured from the agent stream.

    For role="assistant": text is the assistant's text response; tool_calls is [].
    For role="tool_use": tool_calls holds the resolved tool invocations; text is None.
    """

    turn_idx: int
    role: Literal["assistant", "tool_use"]
    text: str | None  # set for role="assistant", None for tool_use
    tool_calls: list[ToolCall]  # set for role="tool_use", [] for assistant
