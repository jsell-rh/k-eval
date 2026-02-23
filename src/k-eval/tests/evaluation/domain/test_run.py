"""Tests for the EvaluationRun domain model."""

import json

import pytest
from pydantic import ValidationError

from agent.domain.result import AgentResult
from agent.domain.usage import UsageMetrics
from dataset.domain.sample import Sample
from evaluation.domain.run import EvaluationRun
from judge.domain.score import JudgeResult


def _make_sample() -> Sample:
    return Sample(
        sample_idx="s1", question="What is k-eval?", answer="An eval framework."
    )


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

        with pytest.raises(ValidationError):
            run.run_id = "changed"  # type: ignore[misc]

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


class TestEvaluationRunConstraints:
    """EvaluationRun rejects invalid field values."""

    def test_empty_run_id_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationRun(
                run_id="",
                sample=_make_sample(),
                condition="baseline",
                run_index=0,
                agent_result=_make_agent_result(),
                judge_result=_make_judge_result(),
            )

    def test_empty_condition_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationRun(
                run_id="run-abc",
                sample=_make_sample(),
                condition="",
                run_index=0,
                agent_result=_make_agent_result(),
                judge_result=_make_judge_result(),
            )

    def test_negative_run_index_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationRun(
                run_id="run-abc",
                sample=_make_sample(),
                condition="baseline",
                run_index=-1,
                agent_result=_make_agent_result(),
                judge_result=_make_judge_result(),
            )

    def test_run_index_zero_is_valid(self) -> None:
        run = EvaluationRun(
            run_id="run-abc",
            sample=_make_sample(),
            condition="baseline",
            run_index=0,
            agent_result=_make_agent_result(),
            judge_result=_make_judge_result(),
        )
        assert run.run_index == 0


class TestEvaluationRunSerialization:
    """EvaluationRun is fully serializable via Pydantic."""

    def test_model_dump_json_succeeds(self) -> None:
        run = EvaluationRun(
            run_id="run-abc",
            sample=_make_sample(),
            condition="baseline",
            run_index=0,
            agent_result=_make_agent_result(),
            judge_result=_make_judge_result(),
        )

        raw = run.model_dump_json()

        assert isinstance(raw, str)
        parsed = json.loads(raw)
        assert parsed["run_id"] == "run-abc"
        assert parsed["condition"] == "baseline"
        assert parsed["run_index"] == 0

    def test_model_dump_json_includes_sample_fields(self) -> None:
        run = EvaluationRun(
            run_id="run-abc",
            sample=_make_sample(),
            condition="baseline",
            run_index=0,
            agent_result=_make_agent_result(),
            judge_result=_make_judge_result(),
        )

        parsed = json.loads(run.model_dump_json())

        assert parsed["sample"]["sample_idx"] == "s1"
        assert parsed["sample"]["question"] == "What is k-eval?"

    def test_model_dump_json_includes_agent_result_fields(self) -> None:
        run = EvaluationRun(
            run_id="run-abc",
            sample=_make_sample(),
            condition="baseline",
            run_index=0,
            agent_result=_make_agent_result(),
            judge_result=_make_judge_result(),
        )

        parsed = json.loads(run.model_dump_json())

        assert parsed["agent_result"]["response"] == "It is an eval framework."
        assert parsed["agent_result"]["usage"]["input_tokens"] == 50
