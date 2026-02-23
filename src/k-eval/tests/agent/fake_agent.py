"""FakeAgent — in-memory Agent implementation for use in tests."""

from agent.domain.result import AgentResult


class FakeAgent:
    """Satisfies the Agent protocol. Returns a canned result for any question."""

    def __init__(self, result: AgentResult) -> None:
        self._result = result

    async def ask(self, question: str) -> AgentResult:
        return self._result
