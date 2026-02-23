"""FakeAgentFactory — in-memory AgentFactory implementation for use in tests."""

from agent.domain.agent import Agent
from config.domain.condition_mcp_server import ConditionMcpServer
from tests.agent.fake_agent import FakeAgent
from agent.domain.result import AgentResult


class FakeAgentFactory:
    """Satisfies the AgentFactory protocol. Returns a FakeAgent for every create call."""

    def __init__(self, result: AgentResult) -> None:
        self._result = result
        self.created: list[dict[str, object]] = []

    def create(
        self,
        condition: str,
        sample_id: str,
        system_prompt: str,
        mcp_servers: list[ConditionMcpServer],
    ) -> Agent:
        self.created.append(
            {
                "condition": condition,
                "sample_id": sample_id,
                "system_prompt": system_prompt,
                "mcp_servers": mcp_servers,
            }
        )
        return FakeAgent(result=self._result)
