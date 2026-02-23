"""FakeJudge — in-memory Judge implementation for use in tests."""

from judge.domain.score import JudgeResult

_DEFAULT_RESULT = JudgeResult(
    factual_adherence=5,
    factual_adherence_reasoning="Accurate.",
    completeness=5,
    completeness_reasoning="Complete.",
    helpfulness_and_clarity=5,
    helpfulness_and_clarity_reasoning="Clear.",
    unverified_claims=[],
)


class FakeJudge:
    """Satisfies the Judge protocol. Returns a canned result for any score call."""

    def __init__(self, result: JudgeResult = _DEFAULT_RESULT) -> None:
        self._result = result
        self.scores_requested: list[dict[str, str]] = []

    async def score(
        self, question: str, golden_answer: str, agent_response: str
    ) -> JudgeResult:
        self.scores_requested.append(
            {
                "question": question,
                "golden_answer": golden_answer,
                "agent_response": agent_response,
            }
        )
        return self._result
