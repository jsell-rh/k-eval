"""Tests for the EvaluationRun domain model."""

from agent.domain.result import AgentResult
from agent.domain.usage import UsageMetrics
from dataset.domain.sample import Sample
from evaluation.domain.run import EvaluationRun
from judge.domain.score import JudgeResult


def _make_sample() -> Sample:
    return Sample(id="s1", question="What is k-eval?", answer="An eval framework.")


def _make_agent_result() -> AgentResult:
    return AgentResult(
        response="It is an eval framework.",
        cost_usd=0.001,
        duration_ms=500,
        duration_api_ms=400,
        num_turns=1,
        usage=UsageMetrics(input_tokens=50, output_tokens=20),
    )


def _make_judge_result() -> JudgeResult:
    return JudgeResult(
        factual_adherence=5,
        factual_adherence_reasoning="Accurate.",
        completeness=4,
        completeness_reasoning="Nearly complete.",
        helpfulness_and_clarity=5,
        helpfulness_and_clarity_reasoning="Very clear.",
        unverified_claims=[],
    )


class TestEvaluationRun:
    def test_construction_succeeds_with_valid_fields(self) -> None:
        run = EvaluationRun(
            run_id="run-abc",
            sample=_make_sample(),
            condition="baseline",
            run_index=0,
            agent_result=_make_agent_result(),
            judge_result=_make_judge_result(),
        )

        assert run.run_id == "run-abc"
        assert run.condition == "baseline"
        assert run.run_index == 0

    def test_sample_is_preserved(self) -> None:
        sample = _make_sample()
        run = EvaluationRun(
            run_id="run-abc",
            sample=sample,
            condition="baseline",
            run_index=0,
            agent_result=_make_agent_result(),
            judge_result=_make_judge_result(),
        )

        assert run.sample is sample

    def test_run_is_immutable(self) -> None:
        run = EvaluationRun(
            run_id="run-abc",
            sample=_make_sample(),
            condition="baseline",
            run_index=0,
            agent_result=_make_agent_result(),
            judge_result=_make_judge_result(),
        )

        try:
            run.run_id = "changed"  # type: ignore[misc]
            raise AssertionError("Expected assignment to raise")
        except Exception:
            pass

    def test_agent_result_is_stored(self) -> None:
        agent_result = _make_agent_result()
        run = EvaluationRun(
            run_id="run-abc",
            sample=_make_sample(),
            condition="baseline",
            run_index=0,
            agent_result=agent_result,
            judge_result=_make_judge_result(),
        )

        assert run.agent_result is agent_result

    def test_judge_result_is_stored(self) -> None:
        judge_result = _make_judge_result()
        run = EvaluationRun(
            run_id="run-abc",
            sample=_make_sample(),
            condition="baseline",
            run_index=0,
            agent_result=_make_agent_result(),
            judge_result=judge_result,
        )

        assert run.judge_result is judge_result

    def test_run_index_reflects_repetition(self) -> None:
        run = EvaluationRun(
            run_id="run-abc",
            sample=_make_sample(),
            condition="with-graph",
            run_index=2,
            agent_result=_make_agent_result(),
            judge_result=_make_judge_result(),
        )

        assert run.run_index == 2
