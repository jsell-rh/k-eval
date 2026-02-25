"""AgentFactory Protocol — structural interface for constructing Agent instances."""

from typing import Protocol

from k_eval.agent.domain.agent import Agent
from k_eval.config.domain.condition_mcp_server import ConditionMcpServer


class AgentFactory(Protocol):
    """Constructs a new Agent instance for a given (condition, sample) pair."""

    def create(
        self,
        condition: str,
        sample_idx: str,
        system_prompt: str,
        mcp_servers: list[ConditionMcpServer],
    ) -> Agent: ...
