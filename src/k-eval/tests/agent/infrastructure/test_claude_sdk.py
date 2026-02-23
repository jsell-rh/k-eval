"""Tests for ClaudeAgentSDKAgent infrastructure implementation."""

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import MagicMock, patch

from claude_agent_sdk.types import (
    McpHttpServerConfig,
    McpSSEServerConfig,
    McpStdioServerConfig,
)

import pytest

from claude_agent_sdk._errors import ClaudeSDKError
from claude_agent_sdk.types import ResultMessage

from agent.infrastructure.claude_sdk import ClaudeAgentSDKAgent
from agent.infrastructure.errors import AgentInvocationError
from config.domain.agent import AgentConfig
from config.domain.condition_mcp_server import ConditionMcpServer
from config.domain.mcp_server import HttpMcpServer, SseMcpServer, StdioMcpServer
from tests.agent.fake_observer import FakeAgentObserver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    mcp_servers: list[ConditionMcpServer] | None = None,
    model: str = "claude-3-5-sonnet-20241022",
    condition: str = "baseline",
    sample_id: str = "0",
    observer: FakeAgentObserver | None = None,
) -> ClaudeAgentSDKAgent:
    return ClaudeAgentSDKAgent(
        config=AgentConfig(type="claude-sdk", model=model),
        condition=condition,
        sample_id=sample_id,
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
# _build_allowed_tools tests
# ---------------------------------------------------------------------------


class TestBuildAllowedTools:
    """_build_allowed_tools() produces correct wildcard tool entries."""

    def test_empty_servers_returns_empty_list(self) -> None:
        agent = _make_agent(mcp_servers=[])

        result = agent._build_allowed_tools()

        assert result == []

    def test_single_server_returns_wildcard_entry(self) -> None:
        agent = _make_agent(mcp_servers=[_stdio_server(name="graph")])

        result = agent._build_allowed_tools()

        assert result == ["mcp__graph__*"]

    def test_multiple_servers_returns_all_wildcard_entries(self) -> None:
        agent = _make_agent(
            mcp_servers=[
                _stdio_server(name="graph"),
                _sse_server(name="search"),
            ]
        )

        result = agent._build_allowed_tools()

        assert result == ["mcp__graph__*", "mcp__search__*"]


# ---------------------------------------------------------------------------
# ask() tests — mock claude_agent_sdk.query
# ---------------------------------------------------------------------------


class TestAskSuccess:
    """ask() returns correct AgentResult on a successful invocation."""

    async def test_returns_agent_result_with_response_text(self) -> None:
        result_msg = _make_result_message(result="The answer is 42.")
        agent = _make_agent()

        with patch(
            "agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            result = await agent.ask(question="What is the answer?")

        assert result.response == "The answer is 42."

    async def test_returns_agent_result_with_cost(self) -> None:
        result_msg = _make_result_message(total_cost_usd=0.005)
        agent = _make_agent()

        with patch(
            "agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            result = await agent.ask(question="What is the answer?")

        assert result.cost_usd == pytest.approx(0.005)

    async def test_returns_agent_result_with_duration(self) -> None:
        result_msg = _make_result_message(duration_ms=1500, duration_api_ms=1200)
        agent = _make_agent()

        with patch(
            "agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            result = await agent.ask(question="What is the answer?")

        assert result.duration_ms == 1500
        assert result.duration_api_ms == 1200

    async def test_returns_agent_result_with_num_turns(self) -> None:
        result_msg = _make_result_message(num_turns=3)
        agent = _make_agent()

        with patch(
            "agent.infrastructure.claude_sdk.query",
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
            "agent.infrastructure.claude_sdk.query",
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
            "agent.infrastructure.claude_sdk.query",
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
            "agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What is the answer?")

    async def test_result_message_is_error_message_starts_with_failed(self) -> None:
        result_msg = _make_result_message(is_error=True, result="Boom.")
        agent = _make_agent()

        with patch(
            "agent.infrastructure.claude_sdk.query",
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
            "agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What is the answer?")

    async def test_no_result_message_in_generator_raises_agent_invocation_error(
        self,
    ) -> None:
        agent = _make_agent()

        with patch(
            "agent.infrastructure.claude_sdk.query",
            new=_mock_query(),  # empty — no ResultMessage
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What is the answer?")

    async def test_sdk_exception_is_wrapped_in_agent_invocation_error(self) -> None:
        agent = _make_agent()

        with patch(
            "agent.infrastructure.claude_sdk.query",
            new=_mock_query_raising(ClaudeSDKError("CLI not found")),
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What is the answer?")

    async def test_sdk_exception_message_starts_with_failed(self) -> None:
        agent = _make_agent()

        with patch(
            "agent.infrastructure.claude_sdk.query",
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
        agent = _make_agent(condition="with-graph", sample_id="7", observer=observer)

        with patch(
            "agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            await agent.ask(question="What?")

        assert len(observer.invocation_started) == 1
        assert observer.invocation_started[0].condition == "with-graph"
        assert observer.invocation_started[0].sample_id == "7"
        assert observer.invocation_started[0].model == "claude-3-5-sonnet-20241022"

    async def test_invocation_completed_emitted_on_success(self) -> None:
        result_msg = _make_result_message(
            duration_ms=1500, num_turns=2, total_cost_usd=0.005
        )
        observer = FakeAgentObserver()
        agent = _make_agent(condition="with-graph", sample_id="7", observer=observer)

        with patch(
            "agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            await agent.ask(question="What?")

        assert len(observer.invocation_completed) == 1
        event = observer.invocation_completed[0]
        assert event.condition == "with-graph"
        assert event.sample_id == "7"
        assert event.duration_ms == 1500
        assert event.num_turns == 2
        assert event.cost_usd == pytest.approx(0.005)

    async def test_invocation_failed_emitted_on_sdk_error(self) -> None:
        observer = FakeAgentObserver()
        agent = _make_agent(condition="with-graph", sample_id="7", observer=observer)

        with patch(
            "agent.infrastructure.claude_sdk.query",
            new=_mock_query_raising(ClaudeSDKError("connection refused")),
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What?")

        assert len(observer.invocation_failed) == 1
        assert observer.invocation_failed[0].condition == "with-graph"
        assert observer.invocation_failed[0].sample_id == "7"

    async def test_invocation_failed_emitted_on_result_error(self) -> None:
        result_msg = _make_result_message(is_error=True, result="agent errored")
        observer = FakeAgentObserver()
        agent = _make_agent(condition="baseline", sample_id="3", observer=observer)

        with patch(
            "agent.infrastructure.claude_sdk.query",
            new=_mock_query(result_msg),
        ):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What?")

        assert len(observer.invocation_failed) == 1

    async def test_invocation_completed_not_emitted_on_error(self) -> None:
        observer = FakeAgentObserver()
        agent = _make_agent(observer=observer)

        with patch(
            "agent.infrastructure.claude_sdk.query",
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
        agent = _make_agent(condition="with-graph", sample_id="9", observer=observer)

        def _raise() -> None:
            raise AgentInvocationError(
                reason="unsupported MCP server type for server 'bad'"
            )

        with patch.object(agent, "_build_mcp_servers", side_effect=_raise):
            with pytest.raises(AgentInvocationError):
                await agent.ask(question="What?")

        assert len(observer.invocation_failed) == 1
        assert observer.invocation_failed[0].condition == "with-graph"
        assert observer.invocation_failed[0].sample_id == "9"
