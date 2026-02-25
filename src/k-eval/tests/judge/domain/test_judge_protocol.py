"""Tests that the Judge Protocol is satisfied by FakeJudge."""

from k_eval.judge.domain.judge import Judge
from tests.judge.fake_judge import FakeJudge


class TestJudgeProtocol:
    """FakeJudge satisfies the Judge structural protocol."""

    def test_fake_judge_satisfies_judge_protocol(self) -> None:
        judge: Judge = FakeJudge()

        # Static check: assignment to Judge-typed variable succeeds at runtime.
        # mypy will also verify this at type-check time.
        assert judge is not None

    async def test_fake_judge_score_returns_judge_result(self) -> None:
        judge = FakeJudge()

        result = await judge.score(
            question="What is the answer?",
            golden_answer="42",
            agent_response="The answer is 42.",
        )

        assert result.factual_adherence == 5
        assert result.completeness == 5
        assert result.helpfulness_and_clarity == 5

    async def test_fake_judge_records_score_call(self) -> None:
        judge = FakeJudge()

        await judge.score(
            question="What is the answer?",
            golden_answer="42",
            agent_response="The answer is 42.",
        )

        assert len(judge.scores_requested) == 1
        assert judge.scores_requested[0]["question"] == "What is the answer?"
        assert judge.scores_requested[0]["golden_answer"] == "42"
        assert judge.scores_requested[0]["agent_response"] == "The answer is 42."

    async def test_fake_judge_records_multiple_score_calls(self) -> None:
        judge = FakeJudge()

        await judge.score(
            question="Q1",
            golden_answer="A1",
            agent_response="R1",
        )
        await judge.score(
            question="Q2",
            golden_answer="A2",
            agent_response="R2",
        )

        assert len(judge.scores_requested) == 2
