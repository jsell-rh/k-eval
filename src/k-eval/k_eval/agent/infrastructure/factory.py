"""ClaudeAgentSDKAgentFactory — constructs ClaudeAgentSDKAgent instances."""

from k_eval.agent.domain.agent import Agent
from k_eval.agent.domain.observer import AgentObserver
from k_eval.agent.infrastructure.claude_sdk import ClaudeAgentSDKAgent
from k_eval.config.domain.agent import AgentConfig
from k_eval.config.domain.condition_mcp_server import ConditionMcpServer


class ClaudeAgentSDKAgentFactory:
    """Creates ClaudeAgentSDKAgent instances configured for a given condition and sample."""

    def __init__(self, config: AgentConfig, observer: AgentObserver) -> None:
        self._config = config
        self._observer = observer

    def create(
        self,
        condition: str,
        sample_idx: str,
        system_prompt: str,
        mcp_servers: list[ConditionMcpServer],
    ) -> Agent:
        """Construct a new ClaudeAgentSDKAgent for the given condition and sample."""
        return ClaudeAgentSDKAgent(
            config=self._config,
            condition=condition,
            sample_idx=sample_idx,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            observer=self._observer,
        )
