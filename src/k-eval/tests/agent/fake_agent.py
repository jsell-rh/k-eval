"""FakeAgent — in-memory Agent implementation for use in tests."""

from k_eval.agent.domain.result import AgentResult


class FakeAgent:
    """Satisfies the Agent protocol. Returns a canned result for any question.

    If side_effects is provided, each call pops from the front of the list:
    - If the item is an Exception, it is raised.
    - If the item is an AgentResult, it is returned.
    Once the list is exhausted, the default result is returned for all subsequent calls.
    """

    def __init__(
        self,
        result: AgentResult,
        side_effects: list[AgentResult | Exception] | None = None,
    ) -> None:
        self._result = result
        self._side_effects: list[AgentResult | Exception] = (
            list(side_effects) if side_effects is not None else []
        )

    async def ask(self, question: str) -> AgentResult:
        if self._side_effects:
            effect = self._side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        return self._result
