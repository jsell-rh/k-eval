"""ClaudeAgentSDKAgent — agent implementation using the Claude Agent SDK."""

from typing import Any

from claude_agent_sdk import query
from claude_agent_sdk._errors import ClaudeSDKError
from claude_agent_sdk.types import (
    ClaudeAgentOptions,
    McpHttpServerConfig,
    McpSdkServerConfig,
    McpSSEServerConfig,
    McpStdioServerConfig,
    ResultMessage,
)

from agent.domain.observer import AgentObserver
from agent.domain.result import AgentResult
from agent.domain.usage import UsageMetrics
from agent.infrastructure.errors import AgentInvocationError
from config.domain.agent import AgentConfig
from config.domain.condition_mcp_server import ConditionMcpServer
from config.domain.mcp_server import HttpMcpServer, SseMcpServer, StdioMcpServer

type McpServerConfigMap = dict[
    str,
    McpStdioServerConfig
    | McpSSEServerConfig
    | McpHttpServerConfig
    | McpSdkServerConfig,
]


class ClaudeAgentSDKAgent:
    """Agent implementation that delegates to the Claude Agent SDK.

    One instance is constructed per (condition, sample) evaluation run.
    The condition and sample_id are injected at construction time so that
    observer events carry full context without polluting the ask() signature.
    """

    def __init__(
        self,
        config: AgentConfig,
        condition: str,
        sample_id: str,
        system_prompt: str,
        mcp_servers: list[ConditionMcpServer],
        observer: AgentObserver,
    ) -> None:
        self._config = config
        self._condition = condition
        self._sample_id = sample_id
        self._system_prompt = system_prompt
        self._mcp_servers = mcp_servers
        self._observer = observer

    async def ask(self, question: str) -> AgentResult:
        """Invoke the agent with a question and return the structured result.

        Opens a new SDK session per call — correct for independent eval samples.

        Raises:
            AgentInvocationError: if the SDK raises, the agent returns an error,
                or no ResultMessage is present in the response stream.
        """
        self._observer.agent_invocation_started(
            condition=self._condition,
            sample_id=self._sample_id,
            model=self._config.model,
        )

        try:
            options = ClaudeAgentOptions(
                model=self._config.model,
                system_prompt=self._system_prompt,
                mcp_servers=self._build_mcp_servers(),
                allowed_tools=self._build_allowed_tools(),
                permission_mode="bypassPermissions",
                setting_sources=[],
            )

            result_message = await self._collect_result(
                prompt=question, options=options
            )
        except AgentInvocationError as exc:
            reason = str(exc).removeprefix("Failed to invoke agent: ")
            self._observer.agent_invocation_failed(
                condition=self._condition,
                sample_id=self._sample_id,
                reason=reason,
            )
            raise

        self._observer.agent_invocation_completed(
            condition=self._condition,
            sample_id=self._sample_id,
            duration_ms=result_message.duration_ms,
            num_turns=result_message.num_turns,
            cost_usd=result_message.total_cost_usd,
        )

        assert result_message.result is not None  # guaranteed by _collect_result
        return AgentResult(
            response=result_message.result,
            cost_usd=result_message.total_cost_usd,
            duration_ms=result_message.duration_ms,
            duration_api_ms=result_message.duration_api_ms,
            num_turns=result_message.num_turns,
            usage=self._map_usage(raw=result_message.usage),
        )

    async def _collect_result(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> ResultMessage:
        """Run the SDK query and extract the single ResultMessage.

        Raises:
            AgentInvocationError: on SDK errors or missing/error ResultMessage.
        """
        result_message: ResultMessage | None = None

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result_message = message
        except ClaudeSDKError as exc:
            raise AgentInvocationError(reason=str(exc)) from exc

        if result_message is None:
            raise AgentInvocationError(reason="no ResultMessage in response stream")

        if result_message.is_error:
            raise AgentInvocationError(
                reason=f"agent returned error response: {result_message.result}"
            )

        if result_message.result is None:
            raise AgentInvocationError(reason="ResultMessage has no result text")

        return result_message

    def _build_mcp_servers(self) -> McpServerConfigMap:
        """Convert ConditionMcpServer list to the SDK's TypedDict format."""
        servers: McpServerConfigMap = {}

        for server in self._mcp_servers:
            config = server.config

            if isinstance(config, StdioMcpServer):
                servers[server.name] = self._build_stdio_server(config=config)
            elif isinstance(config, SseMcpServer):
                servers[server.name] = self._build_sse_server(config=config)
            elif isinstance(config, HttpMcpServer):
                servers[server.name] = self._build_http_server(config=config)
            else:
                raise AgentInvocationError(
                    reason=f"unsupported MCP server type for server '{server.name}'"
                )

        return servers

    def _build_stdio_server(self, config: StdioMcpServer) -> McpStdioServerConfig:
        """Build a McpStdioServerConfig TypedDict from a StdioMcpServer model."""
        server: McpStdioServerConfig = McpStdioServerConfig(command=config.command)
        if config.args:
            server["args"] = list(config.args)
        if config.env:
            server["env"] = dict(config.env)
        return server

    def _build_sse_server(self, config: SseMcpServer) -> McpSSEServerConfig:
        """Build a McpSSEServerConfig TypedDict from a SseMcpServer model."""
        server: McpSSEServerConfig = McpSSEServerConfig(type="sse", url=config.url)
        if config.headers:
            server["headers"] = dict(config.headers)
        return server

    def _build_http_server(self, config: HttpMcpServer) -> McpHttpServerConfig:
        """Build a McpHttpServerConfig TypedDict from an HttpMcpServer model."""
        server: McpHttpServerConfig = McpHttpServerConfig(type="http", url=config.url)
        if config.headers:
            server["headers"] = dict(config.headers)
        return server

    def _build_allowed_tools(self) -> list[str]:
        """Build the allowed tools list — one wildcard entry per MCP server.

        Returns an empty list when there are no servers, which blocks all tools
        (correct for baseline conditions with no MCP context).
        """
        return [f"mcp__{server.name}__*" for server in self._mcp_servers]

    def _map_usage(self, raw: dict[str, Any] | None) -> UsageMetrics | None:
        """Map the SDK's raw usage dict to a typed UsageMetrics value object."""
        if raw is None:
            return None
        return UsageMetrics(
            input_tokens=raw.get("input_tokens"),
            output_tokens=raw.get("output_tokens"),
        )
