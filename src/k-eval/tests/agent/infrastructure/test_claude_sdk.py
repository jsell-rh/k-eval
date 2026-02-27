"""Tests for ClaudeAgentSDKAgent infrastructure implementation."""

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import MagicMock, patch

from claude_agent_sdk.types import (
    AssistantMessage,
    McpHttpServerConfig,
    McpSSEServerConfig,
    McpStdioServerConfig,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

import pytest

from claude_agent_sdk._errors import ClaudeSDKError
from claude_agent_sdk.types import ResultMessage

from k_eval.agent.infrastructure.claude_sdk import ClaudeAgentSDKAgent
from k_eval.agent.infrastructure.errors import AgentInvocationError
from k_eval.config.domain.agent import AgentConfig
from k_eval.config.domain.condition_mcp_server import ConditionMcpServer
from k_eval.config.domain.mcp_server import HttpMcpServer, SseMcpServer, StdioMcpServer
from tests.agent.fake_observer import FakeAgentObserver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    mcp_servers: list[ConditionMcpServer] | None = None,
    model: str = "claude-3-5-sonnet-20241022",
    condition: str = "baseline",
    sample_idx: str = "0",
    observer: FakeAgentObserver | None = None,
) -> ClaudeAgentSDKAgent:
    return ClaudeAgentSDKAgent(
        config=AgentConfig(type="claude-sdk", model=model),
        condition=condition,
        sample_idx=sample_idx,
        system_prompt="You are a helpful assistant.",
        mcp_servers=mcp_servers if mcp_servers is not None else [],
        observer=observer if observer is not None else FakeAgentObserver(),
    )


def _stdio_server(
    name: str = "graph",
    command: str = "uvx",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> ConditionMcpServer:
    return ConditionMcpServer(
        name=name,
        config=StdioMcpServer(
            type="stdio",
            command=command,
            args=args if args is not None else [],
            env=env if env is not None else {},
        ),
    )


def _sse_server(
    name: str = "search",
    url: str = "http://localhost:8080/sse",
    headers: dict[str, str] | None = None,
) -> ConditionMcpServer:
    return ConditionMcpServer(
        name=name,
        config=SseMcpServer(
            type="sse",
            url=url,
            headers=headers if headers is not None else {},
        ),
    )


def _http_server(
    name: str = "api",
    url: str = "http://localhost:9090/mcp",
    headers: dict[str, str] | None = None,
) -> ConditionMcpServer:
    return ConditionMcpServer(
        name=name,
        config=HttpMcpServer(
            type="http",
            url=url,
            headers=headers if headers is not None else {},
        ),
    )


def _make_result_message(
    result: str | None = "The answer is 42.",
    is_error: bool = False,
    duration_ms: int = 1500,
    duration_api_ms: int = 1200,
    num_turns: int = 2,
    total_cost_usd: float | None = 0.005,
    usage: dict[str, Any] | None = None,
) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=duration_ms,
        duration_api_ms=duration_api_ms,
        is_error=is_error,
        num_turns=num_turns,
        session_id="test-session-id",
        total_cost_usd=total_cost_usd,
        usage=usage,
        result=result,
    )


async def _async_gen(*items: Any) -> AsyncIterator[Any]:
    """Async generator yielding a fixed set of items."""
    for item in items:
        yield item


def _mock_query(*items: Any) -> MagicMock:
    """Return a MagicMock for `query` that yields the given items when iterated.

    Since `query` is an async generator function (not a coroutine), patching it
    with a regular Mock whose return_value is an async iterable is the correct
    approach — `mock(...)` returns the async generator directly, which is then
    iterated with `async for`.
    """
    mock = MagicMock()
    mock.return_value = _async_gen(*items)
    return mock


def _mock_query_raising(exc: Exception) -> MagicMock:
    """Return a MagicMock for `query` that raises on iteration."""

    async def _raising_gen() -> AsyncIterator[Any]:
        raise exc
        yield  # make it an async generator

    mock = MagicMock()
    mock.return_value = _raising_gen()
    return mock


# ---------------------------------------------------------------------------
# _build_mcp_servers tests
# ---------------------------------------------------------------------------


class TestBuildMcpServers:
    """_build_mcp_servers() produces correct TypedDict shapes."""

    def test_empty_list_returns_empty_dict(self) -> None:
        agent = _make_agent(mcp_servers=[])

        result = agent._build_mcp_servers()

        assert result == {}

    def test_stdio_server_has_command(self) -> None:
        agent = _make_agent(mcp_servers=[_stdio_server(name="graph", command="uvx")])

        result = cast(McpStdioServerConfig, agent._build_mcp_servers()["graph"])

        assert result["command"] == "uvx"

    def test_stdio_server_args_omitted_when_empty(self) -> None:
        agent = _make_agent(mcp_servers=[_stdio_server(name="graph", args=[])])

        result = cast(McpStdioServerConfig, agent._build_mcp_servers()["graph"])

        assert "args" not in result

    def test_stdio_server_args_present_when_non_empty(self) -> None:
        agent = _make_agent(
            mcp_servers=[_stdio_server(name="graph", args=["run", "my-server"])]
        )

        result = cast(McpStdioServerConfig, agent._build_mcp_servers()["graph"])

        assert result.get("args") == ["run", "my-server"]

    def test_stdio_server_env_omitted_when_empty(self) -> None:
        agent = _make_agent(mcp_servers=[_stdio_server(name="graph", env={})])

        result = cast(McpStdioServerConfig, agent._build_mcp_servers()["graph"])

        assert "env" not in result

    def test_stdio_server_env_present_when_non_empty(self) -> None:
        agent = _make_agent(
            mcp_servers=[_stdio_server(name="graph", env={"FOO": "bar"})]
        )

        result = cast(McpStdioServerConfig, agent._build_mcp_servers()["graph"])

        assert result.get("env") == {"FOO": "bar"}

    def test_sse_server_has_type_and_url(self) -> None:
        agent = _make_agent(
            mcp_servers=[_sse_server(name="search", url="http://localhost:8080/sse")]
        )

        result = cast(McpSSEServerConfig, agent._build_mcp_servers()["search"])

        assert result["type"] == "sse"
        assert result["url"] == "http://localhost:8080/sse"

    def test_sse_server_headers_omitted_when_empty(self) -> None:
        agent = _make_agent(mcp_servers=[_sse_server(name="search", headers={})])

        result = cast(McpSSEServerConfig, agent._build_mcp_servers()["search"])

        assert "headers" not in result

    def test_sse_server_headers_present_when_non_empty(self) -> None:
        agent = _make_agent(
            mcp_servers=[
                _sse_server(name="search", headers={"Authorization": "Bearer tok"})
            ]
        )

        result = cast(McpSSEServerConfig, agent._build_mcp_servers()["search"])

        assert result.get("headers") == {"Authorization": "Bearer tok"}

    def test_http_server_has_type_and_url(self) -> None:
        agent = _make_agent(
            mcp_servers=[_http_server(name="api", url="http://localhost:9090/mcp")]
        )

        result = cast(McpHttpServerConfig, agent._build_mcp_servers()["api"])

        assert result["type"] == "http"
        assert result["url"] == "http://localhost:9090/mcp"

    def test_http_server_headers_omitted_when_empty(self) -> None:
        agent = _make_agent(mcp_servers=[_http_server(name="api", headers={})])

        result = cast(McpHttpServerConfig, agent._build_mcp_servers()["api"])

        assert "headers" not in result

    def test_http_server_headers_present_when_non_empty(self) -> None:
        agent = _make_agent(
            mcp_servers=[_http_server(name="api", headers={"X-Token": "secret"})]
        )

        result = cast(McpHttpServerConfig, agent._build_mcp_servers()["api"])

        assert result.get("headers") == {"X-Token": "secret"}

    def test_mixed_servers_both_present(self) -> None:
        agent = _make_agent(
            mcp_servers=[
                _stdio_server(name="graph"),
                _sse_server(name="search"),
            ]
        )

        result = agent._build_mcp_servers()

        assert "graph" in result
        assert "search" in result
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _build_disallowed_tools tests
# ---------------------------------------------------------------------------


class TestBuildDisallowedTools:
    """_build_disallowed_tools() blocks all Claude built-in tools.

    MCP tool whitelisting via allowed_tools wildcards (mcp__server__*) does not
    work reliably in the SDK. Instead, we rely entirely on disallowed_tools to
    remove all built-in tools, leaving only MCP tools available by default.
    """

    _EXPECTED_BUILTINS = [
        "Bash",
        "Edit",
        "Glob",
        "Grep",
        "LS",
        "MultiEdit",
        "NotebookEdit",
        "NotebookRead",
        "Read",
        "Task",
        "TodoRead",
        "TodoWrite",
        "WebFetch",
        "WebSearch",
        "Write",
    ]

    def test_returns_all_builtin_tools(self) -> None:
        agent = _make_agent()

        result = agent._build_disallowed_tools()

        assert result == self._EXPECTED_BUILTINS

    def test_same_regardless_of_mcp_servers(self) -> None:
        agent_no_servers = _make_agent(mcp_servers=[])
        agent_with_servers = _make_agent(
            mcp_servers=[_stdio_server(name="graph"), _sse_server(name="search")]
        )

        assert (
            agent_no_servers._build_disallowed_tools()
            == agent_with_servers._build_disallowed_tools()
        )

    def test_websearch_is_blocked(self) -> None:
        result = _make_agent()._build_disallowed_tools()

        assert "WebSearch" in result

    def test_bash_is_blocked(self) -> None:
        result = _make_agent()._build_disallowed_tools()

        assert "Bash" in result

    def test_webfetch_is_blocked(self) -> None:
        result = _make_agent()._build_disallowed_tools()

        assert "WebFetch" in result


# ---------------------------------------------------------------------------
# ask() tests — mock claude_agent_sdk.query
# ---------------------------------------------------------------------------


class TestAskSuccess:
    """ask() returns correct AgentResult on a successful invocation."""

    async def test_returns_agent_result_with_response_text(self) -> None:
        result_msg = _make_result_message(result="The answer is 42.")
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            result = await agent.ask(question="What is the answer?")

        assert result.response == "The answer is 42."

    async def test_returns_agent_result_with_cost(self) -> None:
        result_msg = _make_result_message(total_cost_usd=0.005)
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            result = await agent.ask(question="What is the answer?")

        assert result.cost_usd == pytest.approx(0.005)

    async def test_returns_agent_result_with_duration(self) -> None:
        result_msg = _make_result_message(duration_ms=1500, duration_api_ms=1200)
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            result = await agent.ask(question="What is the answer?")

        assert result.duration_ms == 1500
        assert result.duration_api_ms == 1200

    async def test_returns_agent_result_with_num_turns(self) -> None:
        result_msg = _make_result_message(num_turns=3)
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            result = await agent.ask(question="What is the answer?")

        assert result.num_turns == 3

    async def test_returns_agent_result_with_usage_metrics(self) -> None:
        result_msg = _make_result_message(
            usage={"input_tokens": 100, "output_tokens": 50}
        )
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            result = await agent.ask(question="What is the answer?")

        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50

    async def test_none_usage_maps_to_none(self) -> None:
        result_msg = _make_result_message(usage=None)
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            result = await agent.ask(question="What is the answer?")

        assert result.usage is None


class TestAskErrors:
    """ask() raises AgentInvocationError on various failure conditions."""

    async def test_result_message_is_error_raises_agent_invocation_error(
        self,
    ) -> None:
        result_msg = _make_result_message(is_error=True, result="Something went wrong.")
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What is the answer?")

    async def test_result_message_is_error_message_starts_with_failed(self) -> None:
        result_msg = _make_result_message(is_error=True, result="Boom.")
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            with pytest.raises(AgentInvocationError) as exc_info:
                await agent.ask(question="What is the answer?")

        assert str(exc_info.value).startswith("Failed to ")

    async def test_result_message_with_none_result_raises_agent_invocation_error(
        self,
    ) -> None:
        result_msg = _make_result_message(result=None, is_error=False)
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What is the answer?")

    async def test_no_result_message_in_generator_raises_agent_invocation_error(
        self,
    ) -> None:
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(),  # empty — no ResultMessage
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What is the answer?")

    async def test_sdk_exception_is_wrapped_in_agent_invocation_error(self) -> None:
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query_raising(ClaudeSDKError("CLI not found")),
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What is the answer?")

    async def test_sdk_exception_message_starts_with_failed(self) -> None:
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query_raising(ClaudeSDKError("CLI not found")),
        ):
            with pytest.raises(AgentInvocationError) as exc_info:
                await agent.ask(question="What is the answer?")

        assert str(exc_info.value).startswith("Failed to ")


# ---------------------------------------------------------------------------
# Observer event tests
# ---------------------------------------------------------------------------


class TestObserverEvents:
    """ask() emits the correct observer events on success and failure."""

    async def test_invocation_started_emitted_before_query(self) -> None:
        result_msg = _make_result_message()
        observer = FakeAgentObserver()
        agent = _make_agent(condition="with-graph", sample_idx="7", observer=observer)

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            await agent.ask(question="What?")

        assert len(observer.invocation_started) == 1
        assert observer.invocation_started[0].condition == "with-graph"
        assert observer.invocation_started[0].sample_idx == "7"
        assert observer.invocation_started[0].model == "claude-3-5-sonnet-20241022"

    async def test_invocation_completed_emitted_on_success(self) -> None:
        result_msg = _make_result_message(
            duration_ms=1500, num_turns=2, total_cost_usd=0.005
        )
        observer = FakeAgentObserver()
        agent = _make_agent(condition="with-graph", sample_idx="7", observer=observer)

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            await agent.ask(question="What?")

        assert len(observer.invocation_completed) == 1
        event = observer.invocation_completed[0]
        assert event.condition == "with-graph"
        assert event.sample_idx == "7"
        assert event.duration_ms == 1500
        assert event.num_turns == 2
        assert event.cost_usd == pytest.approx(0.005)

    async def test_invocation_failed_emitted_on_sdk_error(self) -> None:
        observer = FakeAgentObserver()
        agent = _make_agent(condition="with-graph", sample_idx="7", observer=observer)

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query_raising(ClaudeSDKError("connection refused")),
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What?")

        assert len(observer.invocation_failed) == 1
        assert observer.invocation_failed[0].condition == "with-graph"
        assert observer.invocation_failed[0].sample_idx == "7"

    async def test_invocation_failed_emitted_on_result_error(self) -> None:
        result_msg = _make_result_message(is_error=True, result="agent errored")
        observer = FakeAgentObserver()
        agent = _make_agent(condition="baseline", sample_idx="3", observer=observer)

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What?")

        assert len(observer.invocation_failed) == 1

    async def test_invocation_completed_not_emitted_on_error(self) -> None:
        observer = FakeAgentObserver()
        agent = _make_agent(observer=observer)

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query_raising(ClaudeSDKError("timeout")),
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What?")

        assert len(observer.invocation_completed) == 0

    async def test_invocation_failed_emitted_when_build_mcp_servers_raises(
        self,
    ) -> None:
        """Observer event is emitted even when _build_mcp_servers raises (Fix 2)."""
        observer = FakeAgentObserver()
        agent = _make_agent(condition="with-graph", sample_idx="9", observer=observer)

        def _raise() -> None:
            raise AgentInvocationError(
                reason="unsupported MCP server type for server 'bad'"
            )

        with patch.object(agent, "_build_mcp_servers", side_effect=_raise):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What?")

        assert len(observer.invocation_failed) == 1
        assert observer.invocation_failed[0].condition == "with-graph"
        assert observer.invocation_failed[0].sample_idx == "9"


# ---------------------------------------------------------------------------
# Turn collection tests
# ---------------------------------------------------------------------------


def _make_assistant_message_text(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)],
        model="claude-3-5-sonnet-20241022",
    )


def _make_assistant_message_tool_use(
    tool_use_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
) -> AssistantMessage:
    return AssistantMessage(
        content=[ToolUseBlock(id=tool_use_id, name=tool_name, input=tool_input)],
        model="claude-3-5-sonnet-20241022",
    )


def _make_assistant_message_text_and_tool_use(
    text: str,
    tool_use_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
) -> AssistantMessage:
    return AssistantMessage(
        content=[
            TextBlock(text=text),
            ToolUseBlock(id=tool_use_id, name=tool_name, input=tool_input),
        ],
        model="claude-3-5-sonnet-20241022",
    )


def _make_user_message_tool_result(
    tool_use_id: str,
    content: str,
    is_error: bool = False,
) -> UserMessage:
    return UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id=tool_use_id,
                content=content,
                is_error=is_error,
            )
        ]
    )


class TestTurnCollection:
    """ask() populates AgentResult.turns from the SDK message stream."""

    async def test_no_turns_when_only_result_message(self) -> None:
        result_msg = _make_result_message(result="The answer.")
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            result = await agent.ask(question="What?")

        assert result.turns == []

    async def test_assistant_text_turn_captured(self) -> None:
        assistant_msg = _make_assistant_message_text("Here is my reasoning.")
        result_msg = _make_result_message(result="Final answer.")
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(assistant_msg, result_msg),
        ):
            result = await agent.ask(question="What?")

        assert len(result.turns) == 1
        turn = result.turns[0]
        assert turn.role == "assistant"
        assert turn.text == "Here is my reasoning."
        assert turn.tool_calls == []
        assert turn.turn_idx == 0

    async def test_tool_use_turn_captured_after_resolution(self) -> None:
        assistant_msg = _make_assistant_message_tool_use(
            tool_use_id="tu-1",
            tool_name="search_tool",
            tool_input={"query": "climate change"},
        )
        user_msg = _make_user_message_tool_result(
            tool_use_id="tu-1",
            content="Search results here.",
        )
        result_msg = _make_result_message(result="Final answer.")
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(assistant_msg, user_msg, result_msg),
        ):
            result = await agent.ask(question="What?")

        tool_use_turns = [t for t in result.turns if t.role == "tool_use"]
        assert len(tool_use_turns) == 1
        turn = tool_use_turns[0]
        assert len(turn.tool_calls) == 1
        tc = turn.tool_calls[0]
        assert tc.tool_use_id == "tu-1"
        assert tc.tool_name == "search_tool"
        assert tc.tool_input == {"query": "climate change"}
        assert tc.tool_result == "Search results here."
        assert tc.tool_error is False

    async def test_tool_error_captured(self) -> None:
        assistant_msg = _make_assistant_message_tool_use(
            tool_use_id="tu-err",
            tool_name="broken_tool",
            tool_input={},
        )
        user_msg = _make_user_message_tool_result(
            tool_use_id="tu-err",
            content="Connection refused.",
            is_error=True,
        )
        result_msg = _make_result_message(result="Final.")
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(assistant_msg, user_msg, result_msg),
        ):
            result = await agent.ask(question="What?")

        tool_use_turns = [t for t in result.turns if t.role == "tool_use"]
        tc = tool_use_turns[0].tool_calls[0]
        assert tc.tool_error is True
        assert tc.tool_result == "Connection refused."

    async def test_assistant_and_tool_turns_both_captured(self) -> None:
        assistant_msg = _make_assistant_message_text_and_tool_use(
            text="Let me search for that.",
            tool_use_id="tu-2",
            tool_name="lookup",
            tool_input={"term": "MCP"},
        )
        user_msg = _make_user_message_tool_result(
            tool_use_id="tu-2",
            content="MCP stands for Model Context Protocol.",
        )
        result_msg = _make_result_message(result="MCP is a protocol.")
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(assistant_msg, user_msg, result_msg),
        ):
            result = await agent.ask(question="What?")

        # Should have an assistant turn and a tool_use turn
        assistant_turns = [t for t in result.turns if t.role == "assistant"]
        tool_use_turns = [t for t in result.turns if t.role == "tool_use"]
        assert len(assistant_turns) == 1
        assert assistant_turns[0].text == "Let me search for that."
        assert len(tool_use_turns) == 1
        assert tool_use_turns[0].tool_calls[0].tool_name == "lookup"

    async def test_turn_indices_increment(self) -> None:
        assistant_msg = _make_assistant_message_text_and_tool_use(
            text="Searching...",
            tool_use_id="tu-3",
            tool_name="search",
            tool_input={"q": "test"},
        )
        user_msg = _make_user_message_tool_result(
            tool_use_id="tu-3",
            content="Found: test result.",
        )
        result_msg = _make_result_message(result="Done.")
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(assistant_msg, user_msg, result_msg),
        ):
            result = await agent.ask(question="What?")

        indices = [t.turn_idx for t in result.turns]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)  # all unique

    async def test_unresolved_tool_call_emitted_as_error(self) -> None:
        """A ToolUseBlock with no matching ToolResultBlock becomes tool_error=True."""
        assistant_msg = _make_assistant_message_tool_use(
            tool_use_id="tu-orphan",
            tool_name="orphan_tool",
            tool_input={},
        )
        # No UserMessage with tool result — tool call is never resolved.
        result_msg = _make_result_message(result="Done.")
        agent = _make_agent()

        with patch(
            "k_eval.agent.infrastructure.claude_sdk.query",
            new=_mock_query(assistant_msg, result_msg),
        ):
            result = await agent.ask(question="What?")

        tool_use_turns = [t for t in result.turns if t.role == "tool_use"]
        assert len(tool_use_turns) == 1
        tc = tool_use_turns[0].tool_calls[0]
        assert tc.tool_use_id == "tu-orphan"
        assert tc.tool_error is True
        assert tc.tool_result is None


# ---------------------------------------------------------------------------
# Turn collection — duration tests
# ---------------------------------------------------------------------------


class TestTurnCollectionDuration:
    """Resolved tool calls carry duration_ms; unresolved calls have duration_ms=None."""

    async def test_resolved_tool_call_has_duration_ms(self) -> None:
        """A resolved tool call has duration_ms set to a positive float."""
        assistant_msg = _make_assistant_message_tool_use(
            tool_use_id="tu-timed",
            tool_name="timed_tool",
            tool_input={"q": "test"},
        )
        user_msg = _make_user_message_tool_result(
            tool_use_id="tu-timed",
            content="Result.",
        )
        result_msg = _make_result_message(result="Done.")
        agent = _make_agent()

        # Fake monotonic time: first call returns 100.0 (start), second returns 101.5 (end).
        time_values = [100.0, 101.5]
        call_count = 0

        def fake_monotonic() -> float:
            nonlocal call_count
            val = time_values[call_count]
            call_count += 1
            return val

        with (
            patch(
                "k_eval.agent.infrastructure.claude_sdk.query",
                new=_mock_query(assistant_msg, user_msg, result_msg),
            ),
            patch(
                "k_eval.agent.infrastructure.claude_sdk.time.monotonic",
                side_effect=fake_monotonic,
            ),
        ):
            result = await agent.ask(question="What?")

        tool_use_turns = [t for t in result.turns if t.role == "tool_use"]
        assert len(tool_use_turns) == 1
        tc = tool_use_turns[0].tool_calls[0]
        assert tc.duration_ms is not None
        assert abs(tc.duration_ms - 1500.0) < 1.0  # 1.5s * 1000 = 1500ms

    async def test_unresolved_tool_call_has_duration_ms_none(self) -> None:
        """An unresolved tool call (orphan) has duration_ms=None."""
        assistant_msg = _make_assistant_message_tool_use(
            tool_use_id="tu-orphan",
            tool_name="orphan_tool",
            tool_input={},
        )
        result_msg = _make_result_message(result="Done.")
        agent = _make_agent()

        with (
            patch(
                "k_eval.agent.infrastructure.claude_sdk.query",
                new=_mock_query(assistant_msg, result_msg),
            ),
            patch(
                "k_eval.agent.infrastructure.claude_sdk.time.monotonic",
                return_value=100.0,
            ),
        ):
            result = await agent.ask(question="What?")

        tool_use_turns = [t for t in result.turns if t.role == "tool_use"]
        tc = tool_use_turns[0].tool_calls[0]
        assert tc.duration_ms is None

    async def test_multiple_tool_calls_each_have_independent_duration(self) -> None:
        """Two sequential tool calls each carry their own duration_ms."""
        # First tool use + result
        assist1 = _make_assistant_message_tool_use(
            tool_use_id="tu-1",
            tool_name="tool_one",
            tool_input={},
        )
        user1 = _make_user_message_tool_result(tool_use_id="tu-1", content="R1.")
        # Second tool use + result
        assist2 = _make_assistant_message_tool_use(
            tool_use_id="tu-2",
            tool_name="tool_two",
            tool_input={},
        )
        user2 = _make_user_message_tool_result(tool_use_id="tu-2", content="R2.")
        result_msg = _make_result_message(result="Done.")
        agent = _make_agent()

        # Times: tu-1 start=100, end=101 (1000ms); tu-2 start=102, end=104 (2000ms)
        time_values = [100.0, 101.0, 102.0, 104.0]
        call_count = 0

        def fake_monotonic() -> float:
            nonlocal call_count
            val = time_values[call_count]
            call_count += 1
            return val

        with (
            patch(
                "k_eval.agent.infrastructure.claude_sdk.query",
                new=_mock_query(assist1, user1, assist2, user2, result_msg),
            ),
            patch(
                "k_eval.agent.infrastructure.claude_sdk.time.monotonic",
                side_effect=fake_monotonic,
            ),
        ):
            result = await agent.ask(question="What?")

        tool_calls = [
            tc
            for turn in result.turns
            if turn.role == "tool_use"
            for tc in turn.tool_calls
        ]
        assert len(tool_calls) == 2
        dur0 = tool_calls[0].duration_ms
        dur1 = tool_calls[1].duration_ms
        assert dur0 is not None
        assert dur1 is not None
        assert abs(dur0 - 1000.0) < 1.0
        assert abs(dur1 - 2000.0) < 1.0

    async def test_error_tool_call_has_duration_ms(self) -> None:
        """A tool call that errors (is_error=True) still has duration_ms set."""
        assistant_msg = _make_assistant_message_tool_use(
            tool_use_id="tu-err",
            tool_name="failing_tool",
            tool_input={},
        )
        user_msg = _make_user_message_tool_result(
            tool_use_id="tu-err",
            content="Error: connection refused.",
            is_error=True,
        )
        result_msg = _make_result_message(result="Done.")
        agent = _make_agent()

        time_values = [200.0, 200.5]
        call_count = 0

        def fake_monotonic() -> float:
            nonlocal call_count
            val = time_values[call_count]
            call_count += 1
            return val

        with (
            patch(
                "k_eval.agent.infrastructure.claude_sdk.query",
                new=_mock_query(assistant_msg, user_msg, result_msg),
            ),
            patch(
                "k_eval.agent.infrastructure.claude_sdk.time.monotonic",
                side_effect=fake_monotonic,
            ),
        ):
            result = await agent.ask(question="What?")

        tool_calls = [
            tc
            for turn in result.turns
            if turn.role == "tool_use"
            for tc in turn.tool_calls
        ]
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc.tool_error is True
        assert tc.duration_ms is not None
        assert abs(tc.duration_ms - 500.0) < 1.0
