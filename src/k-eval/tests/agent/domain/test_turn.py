"""Tests for ToolCall and AgentTurn domain value objects."""

import pytest
from pydantic import ValidationError

from k_eval.agent.domain.turn import AgentTurn, ToolCall


class TestToolCall:
    """ToolCall is a frozen Pydantic model with required fields."""

    def test_tool_call_is_immutable(self) -> None:
        tc = ToolCall(
            tool_use_id="id-1",
            tool_name="my_tool",
            tool_input={"key": "value"},
            tool_result="result text",
            tool_error=False,
        )

        with pytest.raises((TypeError, ValidationError)):
            tc.tool_name = "changed"  # type: ignore[misc]

    def test_tool_call_with_result_text(self) -> None:
        tc = ToolCall(
            tool_use_id="id-1",
            tool_name="search_tool",
            tool_input={"query": "hello"},
            tool_result="found results",
            tool_error=False,
        )

        assert tc.tool_use_id == "id-1"
        assert tc.tool_name == "search_tool"
        assert tc.tool_input == {"query": "hello"}
        assert tc.tool_result == "found results"
        assert tc.tool_error is False

    def test_tool_call_with_none_result(self) -> None:
        tc = ToolCall(
            tool_use_id="id-2",
            tool_name="broken_tool",
            tool_input={},
            tool_result=None,
            tool_error=True,
        )

        assert tc.tool_result is None
        assert tc.tool_error is True

    def test_tool_call_with_complex_input(self) -> None:
        tc = ToolCall(
            tool_use_id="id-3",
            tool_name="complex_tool",
            tool_input={"nested": {"key": 1}, "list": [1, 2, 3]},
            tool_result="ok",
            tool_error=False,
        )

        assert tc.tool_input["nested"] == {"key": 1}
        assert tc.tool_input["list"] == [1, 2, 3]

    def test_tool_call_requires_tool_use_id(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall(  # type: ignore[call-arg]
                tool_name="my_tool",
                tool_input={},
                tool_result=None,
                tool_error=False,
            )

    def test_tool_call_requires_tool_name(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall(  # type: ignore[call-arg]
                tool_use_id="id-1",
                tool_input={},
                tool_result=None,
                tool_error=False,
            )


class TestAgentTurn:
    """AgentTurn is a frozen Pydantic model with role-specific semantics."""

    def test_assistant_turn_with_text(self) -> None:
        turn = AgentTurn(
            turn_idx=0,
            role="assistant",
            text="Here is my analysis.",
            tool_calls=[],
        )

        assert turn.turn_idx == 0
        assert turn.role == "assistant"
        assert turn.text == "Here is my analysis."
        assert turn.tool_calls == []

    def test_tool_use_turn_with_calls(self) -> None:
        tc = ToolCall(
            tool_use_id="id-1",
            tool_name="search",
            tool_input={"q": "test"},
            tool_result="results",
            tool_error=False,
        )
        turn = AgentTurn(
            turn_idx=1,
            role="tool_use",
            text=None,
            tool_calls=[tc],
        )

        assert turn.turn_idx == 1
        assert turn.role == "tool_use"
        assert turn.text is None
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0] is tc

    def test_agent_turn_is_immutable(self) -> None:
        turn = AgentTurn(
            turn_idx=0,
            role="assistant",
            text="hello",
            tool_calls=[],
        )

        with pytest.raises((TypeError, ValidationError)):
            turn.role = "tool_use"  # type: ignore[misc]

    def test_agent_turn_role_must_be_valid_literal(self) -> None:
        with pytest.raises(ValidationError):
            AgentTurn(
                turn_idx=0,
                role="invalid_role",  # type: ignore[arg-type]
                text="hello",
                tool_calls=[],
            )

    def test_agent_turn_requires_turn_idx(self) -> None:
        with pytest.raises(ValidationError):
            AgentTurn(  # type: ignore[call-arg]
                role="assistant",
                text="hello",
                tool_calls=[],
            )

    def test_tool_use_turn_can_have_multiple_calls(self) -> None:
        calls = [
            ToolCall(
                tool_use_id=f"id-{i}",
                tool_name=f"tool_{i}",
                tool_input={"i": i},
                tool_result=f"result {i}",
                tool_error=False,
            )
            for i in range(3)
        ]
        turn = AgentTurn(
            turn_idx=2,
            role="tool_use",
            text=None,
            tool_calls=calls,
        )

        assert len(turn.tool_calls) == 3

    def test_assistant_turn_with_none_text(self) -> None:
        """text=None is allowed for assistant turns (e.g. thinking-only turn)."""
        turn = AgentTurn(
            turn_idx=0,
            role="assistant",
            text=None,
            tool_calls=[],
        )

        assert turn.text is None


class TestToolCallDuration:
    """ToolCall carries an optional duration_ms field."""

    def test_tool_call_duration_ms_defaults_to_none(self) -> None:
        tc = ToolCall(
            tool_use_id="id-1",
            tool_name="search",
            tool_input={},
            tool_result="ok",
            tool_error=False,
        )

        assert tc.duration_ms is None

    def test_tool_call_duration_ms_can_be_set(self) -> None:
        tc = ToolCall(
            tool_use_id="id-1",
            tool_name="search",
            tool_input={},
            tool_result="ok",
            tool_error=False,
            duration_ms=123.45,
        )

        assert tc.duration_ms == 123.45

    def test_tool_call_duration_ms_is_float(self) -> None:
        tc = ToolCall(
            tool_use_id="id-1",
            tool_name="search",
            tool_input={},
            tool_result="ok",
            tool_error=False,
            duration_ms=500.0,
        )

        assert isinstance(tc.duration_ms, float)

    def test_tool_call_with_error_duration_ms_can_be_none(self) -> None:
        tc = ToolCall(
            tool_use_id="id-err",
            tool_name="broken",
            tool_input={},
            tool_result=None,
            tool_error=True,
            duration_ms=None,
        )

        assert tc.duration_ms is None
