"""FakeAgentFactory — in-memory AgentFactory implementation for use in tests."""

from agent.domain.agent import Agent
from agent.domain.result import AgentResult
from config.domain.condition_mcp_server import ConditionMcpServer
from tests.agent.fake_agent import FakeAgent


class FakeAgentFactory:
    """Satisfies the AgentFactory protocol. Returns a FakeAgent for every create call.

    If agents is provided, each successive create() call pops from the front of the
    list and returns that agent. Once exhausted, a FakeAgent(result=result) is returned
    for all subsequent calls.
    """

    def __init__(
        self,
        result: AgentResult,
        agents: list[FakeAgent] | None = None,
    ) -> None:
        self._result = result
        self._agents: list[FakeAgent] = list(agents) if agents is not None else []
        self.created: list[dict[str, object]] = []

    def create(
        self,
        condition: str,
        sample_idx: str,
        system_prompt: str,
        mcp_servers: list[ConditionMcpServer],
    ) -> Agent:
        self.created.append(
            {
                "condition": condition,
                "sample_idx": sample_idx,
                "system_prompt": system_prompt,
                "mcp_servers": mcp_servers,
            }
        )
        if self._agents:
            return self._agents.pop(0)
        return FakeAgent(result=self._result)
