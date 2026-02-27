"""ClaudeAgentSDKAgent — agent implementation using the Claude Agent SDK."""

import time
from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import query
from claude_agent_sdk._errors import ClaudeSDKError
from claude_agent_sdk.types import (
    AssistantMessage,
    ClaudeAgentOptions,
    McpHttpServerConfig,
    McpSdkServerConfig,
    McpSSEServerConfig,
    McpStdioServerConfig,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from k_eval.agent.domain.observer import AgentObserver
from k_eval.agent.domain.result import AgentResult
from k_eval.agent.domain.turn import AgentTurn, ToolCall
from k_eval.agent.domain.usage import UsageMetrics
from k_eval.agent.infrastructure.errors import AgentInvocationError
from k_eval.config.domain.agent import AgentConfig
from k_eval.config.domain.condition_mcp_server import ConditionMcpServer
from k_eval.config.domain.mcp_server import HttpMcpServer, SseMcpServer, StdioMcpServer


@dataclass(frozen=True)
class _PendingToolCall:
    """Holds a pending ToolCall and the wall-clock start time for duration tracking."""

    tool_call: ToolCall
    start_time: float


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
    The condition and sample_idx are injected at construction time so that
    observer events carry full context without polluting the ask() signature.
    """

    def __init__(
        self,
        config: AgentConfig,
        condition: str,
        sample_idx: str,
        system_prompt: str,
        mcp_servers: list[ConditionMcpServer],
        observer: AgentObserver,
    ) -> None:
        self._config = config
        self._condition = condition
        self._sample_idx = sample_idx
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
            sample_idx=self._sample_idx,
            model=self._config.model,
        )

        try:
            options = ClaudeAgentOptions(
                model=self._config.model,
                system_prompt=self._system_prompt,
                mcp_servers=self._build_mcp_servers(),
                disallowed_tools=self._build_disallowed_tools(),
                permission_mode="bypassPermissions",
                setting_sources=[],
            )

            result_message, turns = await self._collect_result(
                prompt=question, options=options
            )
        except AgentInvocationError as exc:
            reason = str(exc).removeprefix("Failed to invoke agent: ")
            self._observer.agent_invocation_failed(
                condition=self._condition,
                sample_idx=self._sample_idx,
                reason=reason,
            )
            raise

        self._observer.agent_invocation_completed(
            condition=self._condition,
            sample_idx=self._sample_idx,
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
            turns=turns,
        )

    async def _collect_result(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> tuple[ResultMessage, list[AgentTurn]]:
        """Run the SDK query, extract the single ResultMessage, and collect turns.

        Iterates the async message stream from the SDK. For each AssistantMessage,
        text blocks become an assistant turn and tool-use blocks are held as pending
        until a matching ToolResultBlock arrives in a subsequent UserMessage.
        Any pending tool calls that are never resolved are emitted at the end as
        tool_error=True, tool_result=None.

        Raises:
            AgentInvocationError: on SDK errors or missing/error ResultMessage.
        """
        result_message: ResultMessage | None = None
        turns: list[AgentTurn] = []
        turn_idx: int = 0
        # Keyed by tool_use_id; holds _PendingToolCall (ToolCall + start_time)
        # until a ToolResultBlock resolves it.
        pending_tool_calls: dict[str, _PendingToolCall] = {}

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result_message = message
                elif isinstance(message, AssistantMessage):
                    text_parts: list[str] = []
                    tool_uses: list[ToolUseBlock] = []

                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_uses.append(block)

                    # Emit assistant turn if there is text.
                    if text_parts:
                        combined_text = "".join(text_parts)
                        turns.append(
                            AgentTurn(
                                turn_idx=turn_idx,
                                role="assistant",
                                text=combined_text,
                                tool_calls=[],
                            )
                        )
                        turn_idx += 1

                    # Queue pending tool calls, recording start time for duration.
                    for tool_use in tool_uses:
                        pending_tool_calls[tool_use.id] = _PendingToolCall(
                            tool_call=ToolCall(
                                tool_use_id=tool_use.id,
                                tool_name=tool_use.name,
                                tool_input=tool_use.input,
                                tool_result=None,
                                tool_error=False,
                            ),
                            start_time=time.monotonic(),
                        )

                elif isinstance(message, UserMessage):
                    # UserMessage.content may be a str (plain text) or a list of blocks.
                    content = message.content
                    if not isinstance(content, list):
                        continue

                    resolved: list[ToolCall] = []
                    for block in content:
                        if not isinstance(block, ToolResultBlock):
                            continue
                        pending = pending_tool_calls.pop(block.tool_use_id, None)
                        if pending is None:
                            # Result for a tool we didn't track, skip.
                            continue

                        duration_ms = (time.monotonic() - pending.start_time) * 1000.0

                        # content may be str, list-of-dicts, or None.
                        raw_result = block.content
                        if isinstance(raw_result, str):
                            tool_result: str | None = raw_result
                        elif isinstance(raw_result, list):
                            # Extract text from content block dicts.
                            tool_result = " ".join(
                                str(item.get("text", ""))
                                for item in raw_result
                                if isinstance(item, dict)
                            )
                        else:
                            tool_result = None

                        resolved.append(
                            ToolCall(
                                tool_use_id=pending.tool_call.tool_use_id,
                                tool_name=pending.tool_call.tool_name,
                                tool_input=pending.tool_call.tool_input,
                                tool_result=tool_result,
                                tool_error=bool(block.is_error),
                                duration_ms=duration_ms,
                            )
                        )

                    if resolved:
                        turns.append(
                            AgentTurn(
                                turn_idx=turn_idx,
                                role="tool_use",
                                text=None,
                                tool_calls=resolved,
                            )
                        )
                        turn_idx += 1

        except ClaudeSDKError as exc:
            raise AgentInvocationError(reason=str(exc), retriable=True) from exc
        except Exception as exc:
            # The SDK internally raises a bare Exception (not ClaudeSDKError) when
            # its message reader encounters a fatal error (e.g. subprocess exit).
            raise AgentInvocationError(reason=str(exc), retriable=True) from exc

        # Emit any pending tool calls that were never resolved (duration_ms=None).
        if pending_tool_calls:
            unresolved = [
                ToolCall(
                    tool_use_id=p.tool_call.tool_use_id,
                    tool_name=p.tool_call.tool_name,
                    tool_input=p.tool_call.tool_input,
                    tool_result=None,
                    tool_error=True,
                    duration_ms=None,
                )
                for p in pending_tool_calls.values()
            ]
            turns.append(
                AgentTurn(
                    turn_idx=turn_idx,
                    role="tool_use",
                    text=None,
                    tool_calls=unresolved,
                )
            )

        if result_message is None:
            raise AgentInvocationError(reason="no ResultMessage in response stream")

        if result_message.is_error:
            raise AgentInvocationError(
                reason=f"agent returned error response: {result_message.result}"
            )

        if result_message.result is None:
            raise AgentInvocationError(reason="ResultMessage has no result text")

        return result_message, turns

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

    def _build_disallowed_tools(self) -> list[str]:
        """Build the disallowed tools list — all Claude built-in tools.

        allowed_tools alone does not remove built-in tools from the agent's
        context; it only controls approval requirements. Explicitly disallowing
        all built-in tools ensures the agent cannot use web search, file I/O,
        or any other built-in capability regardless of permission_mode.
        """
        return [
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

    def _map_usage(self, raw: dict[str, Any] | None) -> UsageMetrics | None:
        """Map the SDK's raw usage dict to a typed UsageMetrics value object."""
        if raw is None:
            return None
        return UsageMetrics(
            input_tokens=raw.get("input_tokens"),
            output_tokens=raw.get("output_tokens"),
        )
