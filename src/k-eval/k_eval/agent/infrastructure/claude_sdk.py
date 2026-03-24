"""ClaudeAgentSDKAgent — agent implementation using the Claude Agent SDK."""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import structlog

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


def _format_invocation_trace(
    prompt: str,
    turns: list[AgentTurn],
    result_message: ResultMessage | None,
) -> str:
    """Format user_message + assistant/tool turns + final result for failure diagnostics."""
    parts: list[str] = ["--- user_message ---", prompt or "(empty)"]
    for turn in turns:
        if turn.role == "assistant":
            parts.append(f"\n--- ASSISTANT (turn {turn.turn_idx}) ---")
            parts.append(turn.text or "(no text)")
        else:
            parts.append(f"\n--- TOOL_USE (turn {turn.turn_idx}) ---")
            for tc in turn.tool_calls:
                parts.append(f"  {tc.tool_name}: input={tc.tool_input!r}")
                parts.append(f"    result={tc.tool_result!r} error={tc.tool_error}")
    parts.append("\n--- FINAL_RESULT ---")
    if result_message is not None and result_message.result is not None:
        parts.append(result_message.result)
    else:
        parts.append("(none)")
    return "\n".join(parts)
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

    _log = structlog.get_logger()

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

        stderr_lines: list[str] = []

        def _capture_stderr(line: str) -> None:
            stderr_lines.append(line)
            self._log.warning(
                "agent.subprocess.stderr",
                condition=self._condition,
                sample_idx=self._sample_idx,
                line=line,
            )

        try:
            mcp_servers_config = self._build_mcp_servers()
            # SDK expects mcp_servers to be a dict and each config to be a dict (calls .items() on both).
            mcp_servers_config = self._normalize_mcp_servers_for_sdk(mcp_servers_config)
        except AgentInvocationError as exc:
            self._observer.agent_invocation_failed(
                condition=self._condition,
                sample_idx=self._sample_idx,
                reason=str(exc).removeprefix("Failed to invoke agent: "),
            )
            raise

        # Diagnostic: log what MCP server config is being passed
        self._log.debug(
            "agent.mcp_servers_config",
            condition=self._condition,
            sample_idx=self._sample_idx,
            mcp_servers={
                name: {
                    "type": cfg.get("type"),
                    "url": cfg.get("url"),
                    "has_headers": "headers" in cfg,
                    "header_keys": list(cfg.get("headers", {}).keys()),
                }
                for name, cfg in mcp_servers_config.items()
            },
        )

        # Build subprocess environment with explicit credential paths
        subprocess_env = {
            **os.environ,
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
        }
        # Ensure Google ADC is discoverable if using Vertex AI
        if "GOOGLE_APPLICATION_CREDENTIALS" not in subprocess_env:
            adc_path = os.path.expanduser(
                "~/.config/gcloud/application_default_credentials.json"
            )
            if os.path.exists(adc_path):
                subprocess_env["GOOGLE_APPLICATION_CREDENTIALS"] = adc_path

        self._log.debug(
            "agent.subprocess_env",
            condition=self._condition,
            sample_idx=self._sample_idx,
            has_vertex=subprocess_env.get("CLAUDE_CODE_USE_VERTEX"),
            vertex_project=subprocess_env.get("ANTHROPIC_VERTEX_PROJECT_ID"),
            has_gac=bool(subprocess_env.get("GOOGLE_APPLICATION_CREDENTIALS")),
            has_home=bool(subprocess_env.get("HOME")),
        )

        options = ClaudeAgentOptions(
            model=self._config.model,
            system_prompt=self._system_prompt,
            mcp_servers=mcp_servers_config,
            disallowed_tools=self._build_disallowed_tools(),
            permission_mode="bypassPermissions",
            setting_sources=[],
            stderr=_capture_stderr,
            env=subprocess_env,
        )

        try:
            result_message, turns = await self._collect_result(
                prompt=question, options=options
            )
        except AgentInvocationError as exc:
            base_reason = str(exc).removeprefix("Failed to invoke agent: ")
            reason = (
                f"{base_reason}\nstderr: {' | '.join(stderr_lines)}"
                if stderr_lines
                else base_reason
            )
            result_message = getattr(exc, "result_message", None)
            prompt = getattr(exc, "prompt", None) or ""
            turns = getattr(exc, "turns", None) or []
            invocation_trace = _format_invocation_trace(
                prompt=prompt, turns=turns, result_message=result_message
            )
            exc.invocation_trace = invocation_trace  # type: ignore[attr-defined]
            self._log.error(
                "agent.invocation_trace",
                condition=self._condition,
                sample_idx=self._sample_idx,
                full_trace=invocation_trace,
            )
            if result_message is not None:
                result_preview = (
                    (result_message.result or "")[:500]
                    + ("..." if len(result_message.result or "") > 500 else "")
                )
                self._log.info(
                    "agent.result_message_before_failure",
                    condition=self._condition,
                    sample_idx=self._sample_idx,
                    is_error=result_message.is_error,
                    result_preview=result_preview,
                    duration_ms=getattr(result_message, "duration_ms", None),
                    num_turns=getattr(result_message, "num_turns", None),
                )
            else:
                self._log.info(
                    "agent.result_message_before_failure",
                    condition=self._condition,
                    sample_idx=self._sample_idx,
                    result_message=None,
                )
            diagnostic = await self._run_diagnostic(options=options)
            self._log.error(
                "agent.subprocess.diagnostic",
                condition=self._condition,
                sample_idx=self._sample_idx,
                result=diagnostic,
            )
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

    async def _run_diagnostic(self, options: ClaudeAgentOptions) -> str:
        """Spawn the claude CLI directly to capture its actual stdout and stderr.

        The SDK's callback-based stderr capture races with task-group cleanup and
        reliably loses data for fast-failing processes.  Running the subprocess
        ourselves with asyncio.create_subprocess_exec and awaiting communicate()
        guarantees that all output is collected before we return.
        """
        try:
            from claude_agent_sdk._internal.transport.subprocess_cli import (  # noqa: PLC0415
                SubprocessCLITransport,
            )

            transport = SubprocessCLITransport(prompt="", options=options)
            cmd = transport._build_command()

            ping_msg = (
                json.dumps(
                    {
                        "type": "user",
                        "session_id": "",
                        "message": {"role": "user", "content": "ping"},
                        "parent_tool_use_id": None,
                    }
                )
                + "\n"
            )

            proc_env = {
                **os.environ,
                **options.env,
                "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
            }

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(ping_msg.encode()),
                    timeout=10.0,
                )
            except TimeoutError:
                proc.kill()
                return "diagnostic: timed out after 10s (process may be healthy)"

            stdout_text = stdout_bytes.decode(errors="replace").strip()
            stderr_text = stderr_bytes.decode(errors="replace").strip()

            mcp_status = self._extract_mcp_status(stdout_text)

            return (
                f"rc={proc.returncode} | "
                f"mcp_servers={mcp_status} | "
                f"stderr={stderr_text[:2000]!r} | "
                f"stdout={stdout_text[:2000]!r}"
            )
        except Exception as exc:
            return f"diagnostic: could not run subprocess: {exc}"

    def _extract_mcp_status(self, stdout_text: str) -> str:
        """Parse the claude CLI's init JSON from stdout and extract MCP server statuses.

        The init message is the first JSON object on stdout. It contains a
        ``mcp_servers`` list with ``name``, ``status``, and an optional
        ``error`` field for each server.  We surface that directly so the
        diagnostic log line is immediately actionable without reading truncated
        raw stdout.
        """
        for line in stdout_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if msg.get("type") == "system" and msg.get("subtype") == "init":
                servers = msg.get("mcp_servers", [])
                if not servers:
                    return "none"
                parts = []
                for s in servers:
                    name = s.get("name", "?")
                    status = s.get("status", "?")
                    error = s.get("error") or s.get("error_message") or ""
                    parts.append(f"{name}:{status}" + (f"({error})" if error else ""))
                return ", ".join(parts)
        return "init message not found"

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
            # If we already have a valid final result, return it instead of failing.
            # The stream may have died (e.g. MCP timeout) after sending ResultMessage.
            if (
                result_message is not None
                and not result_message.is_error
                and result_message.result is not None
            ):
                self._log.warning(
                    "agent.stream_error_after_result",
                    condition=self._condition,
                    sample_idx=self._sample_idx,
                    error=str(exc),
                )
            else:
                err = AgentInvocationError(reason=str(exc), retriable=True)
                err.result_message = result_message  # type: ignore[attr-defined]
                err.prompt = prompt  # type: ignore[attr-defined]
                err.turns = list(turns)  # type: ignore[attr-defined]
                raise err from exc
        except Exception as exc:
            # The SDK internally raises a bare Exception (not ClaudeSDKError) when
            # its message reader encounters a fatal error (e.g. subprocess exit).
            if (
                result_message is not None
                and not result_message.is_error
                and result_message.result is not None
            ):
                self._log.warning(
                    "agent.stream_error_after_result",
                    condition=self._condition,
                    sample_idx=self._sample_idx,
                    error=str(exc),
                )
            else:
                err = AgentInvocationError(reason=str(exc), retriable=True)
                err.result_message = result_message  # type: ignore[attr-defined]
                err.prompt = prompt  # type: ignore[attr-defined]
                err.turns = list(turns)  # type: ignore[attr-defined]
                raise err from exc

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
            err = AgentInvocationError(reason="no ResultMessage in response stream")
            err.result_message = None  # type: ignore[attr-defined]
            err.prompt = prompt  # type: ignore[attr-defined]
            err.turns = list(turns)  # type: ignore[attr-defined]
            raise err

        if result_message.is_error:
            err = AgentInvocationError(
                reason=f"agent returned error response: {result_message.result}"
            )
            err.result_message = result_message  # type: ignore[attr-defined]
            err.prompt = prompt  # type: ignore[attr-defined]
            err.turns = list(turns)  # type: ignore[attr-defined]
            raise err

        if result_message.result is None:
            err = AgentInvocationError(reason="ResultMessage has no result text")
            err.result_message = result_message  # type: ignore[attr-defined]
            err.prompt = prompt  # type: ignore[attr-defined]
            err.turns = list(turns)  # type: ignore[attr-defined]
            raise err

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

    def _normalize_mcp_servers_for_sdk(
        self, raw: McpServerConfigMap | list[Any]
    ) -> McpServerConfigMap:
        """Ensure the SDK receives a dict of dicts; it calls .items() on mcp_servers and on each config."""
        if isinstance(raw, dict):
            out: McpServerConfigMap = {}
            for name, config in raw.items():
                if isinstance(config, dict):
                    out[name] = config
                else:
                    self._log.warning(
                        "agent.mcp_server_config_skipped",
                        condition=self._condition,
                        sample_idx=self._sample_idx,
                        name=name,
                        reason=f"config must be a dict for SDK, got {type(config).__name__}",
                    )
            return out
        if isinstance(raw, list):
            self._log.warning(
                "agent.mcp_servers_normalized",
                condition=self._condition,
                sample_idx=self._sample_idx,
                reason="mcp_servers was a list; converted to dict for SDK",
            )
            out = {}
            for i, item in enumerate(raw):
                if isinstance(item, dict):
                    name = item.get("name", str(i))
                    out[name] = item
            return out
        return {}

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
