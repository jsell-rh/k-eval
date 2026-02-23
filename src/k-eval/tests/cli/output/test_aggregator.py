"""Tests for cli/output/aggregator.py — aggregate() function."""

import math

import pytest

from agent.domain.result import AgentResult
from agent.domain.usage import UsageMetrics
from cli.output.aggregator import aggregate
from dataset.domain.sample import Sample
from evaluation.domain.run import EvaluationRun
from judge.domain.score import JudgeResult


def _make_sample(idx: str) -> Sample:
    return Sample(
        sample_idx=idx,
        question=f"Question {idx}?",
        answer=f"Answer {idx}.",
    )


def _make_agent_result(response: str = "Agent response.") -> AgentResult:
    return AgentResult(
        response=response,
        cost_usd=0.001,
        duration_ms=500,
        duration_api_ms=400,
        num_turns=1,
        usage=UsageMetrics(input_tokens=50, output_tokens=20),
    )


def _make_judge_result(
    factual_adherence: int = 4,
    completeness: int = 3,
    helpfulness_and_clarity: int = 5,
    unverified_claims: list[str] | None = None,
) -> JudgeResult:
    return JudgeResult(
        factual_adherence=factual_adherence,
        factual_adherence_reasoning="Good.",
        completeness=completeness,
        completeness_reasoning="OK.",
        helpfulness_and_clarity=helpfulness_and_clarity,
        helpfulness_and_clarity_reasoning="Clear.",
        unverified_claims=unverified_claims or [],
    )


def _make_run(
    run_id: str,
    sample: Sample,
    condition: str,
    run_index: int,
    judge_result: JudgeResult | None = None,
    agent_result: AgentResult | None = None,
) -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id,
        sample=sample,
        condition=condition,
        run_index=run_index,
        agent_result=agent_result or _make_agent_result(),
        judge_result=judge_result or _make_judge_result(),
    )


class TestAggregateResultCount:
    """aggregate() produces one AggregatedResult per (sample, condition) pair."""

    def test_two_samples_two_conditions_two_runs_produces_four_results(self) -> None:
        s0 = _make_sample(idx="0")
        s1 = _make_sample(idx="1")
        runs = [
            _make_run(run_id="r", sample=s0, condition="baseline", run_index=0),
            _make_run(run_id="r", sample=s0, condition="baseline", run_index=1),
            _make_run(run_id="r", sample=s0, condition="with-graph", run_index=0),
            _make_run(run_id="r", sample=s0, condition="with-graph", run_index=1),
            _make_run(run_id="r", sample=s1, condition="baseline", run_index=0),
            _make_run(run_id="r", sample=s1, condition="baseline", run_index=1),
            _make_run(run_id="r", sample=s1, condition="with-graph", run_index=0),
            _make_run(run_id="r", sample=s1, condition="with-graph", run_index=1),
        ]

        results = aggregate(runs=runs)

        assert len(results) == 4

    def test_single_sample_single_condition_single_run_produces_one_result(
        self,
    ) -> None:
        s0 = _make_sample(idx="0")
        runs = [
            _make_run(run_id="r", sample=s0, condition="baseline", run_index=0),
        ]

        results = aggregate(runs=runs)

        assert len(results) == 1

    def test_each_result_groups_all_runs_for_that_pair(self) -> None:
        s0 = _make_sample(idx="0")
        runs = [
            _make_run(run_id="r", sample=s0, condition="baseline", run_index=0),
            _make_run(run_id="r", sample=s0, condition="baseline", run_index=1),
        ]

        results = aggregate(runs=runs)

        assert len(results[0].runs) == 2


class TestAggregateMeanAndStddev:
    """Means and stddevs are computed correctly across N runs."""

    def test_mean_is_correct_for_two_runs(self) -> None:
        s0 = _make_sample(idx="0")
        runs = [
            _make_run(
                run_id="r",
                sample=s0,
                condition="baseline",
                run_index=0,
                judge_result=_make_judge_result(
                    factual_adherence=3,
                    completeness=4,
                    helpfulness_and_clarity=5,
                ),
            ),
            _make_run(
                run_id="r",
                sample=s0,
                condition="baseline",
                run_index=1,
                judge_result=_make_judge_result(
                    factual_adherence=5,
                    completeness=2,
                    helpfulness_and_clarity=3,
                ),
            ),
        ]

        results = aggregate(runs=runs)
        result = results[0]

        assert result.factual_adherence_mean == pytest.approx(4.0)
        assert result.completeness_mean == pytest.approx(3.0)
        assert result.helpfulness_and_clarity_mean == pytest.approx(4.0)

    def test_stddev_is_correct_for_two_runs(self) -> None:
        s0 = _make_sample(idx="0")
        runs = [
            _make_run(
                run_id="r",
                sample=s0,
                condition="baseline",
                run_index=0,
                judge_result=_make_judge_result(factual_adherence=3),
            ),
            _make_run(
                run_id="r",
                sample=s0,
                condition="baseline",
                run_index=1,
                judge_result=_make_judge_result(factual_adherence=5),
            ),
        ]

        results = aggregate(runs=runs)
        result = results[0]

        # stdev of [3, 5] = sqrt(2) ≈ 1.4142
        assert result.factual_adherence_stddev == pytest.approx(math.sqrt(2), rel=1e-5)

    def test_stddev_is_zero_for_single_run(self) -> None:
        s0 = _make_sample(idx="0")
        runs = [
            _make_run(
                run_id="r",
                sample=s0,
                condition="baseline",
                run_index=0,
                judge_result=_make_judge_result(factual_adherence=4),
            ),
        ]

        results = aggregate(runs=runs)
        result = results[0]

        assert result.factual_adherence_stddev == 0.0
        assert result.completeness_stddev == 0.0
        assert result.helpfulness_and_clarity_stddev == 0.0

    def test_mean_is_correct_with_identical_scores(self) -> None:
        s0 = _make_sample(idx="0")
        runs = [
            _make_run(
                run_id="r",
                sample=s0,
                condition="baseline",
                run_index=i,
                judge_result=_make_judge_result(factual_adherence=4),
            )
            for i in range(3)
        ]

        results = aggregate(runs=runs)
        result = results[0]

        assert result.factual_adherence_mean == pytest.approx(4.0)
        assert result.factual_adherence_stddev == pytest.approx(0.0)


class TestAggregateUnverifiedClaims:
    """unverified_claims are deduplicated across all runs."""

    def test_claims_are_deduplicated_across_runs(self) -> None:
        s0 = _make_sample(idx="0")
        runs = [
            _make_run(
                run_id="r",
                sample=s0,
                condition="baseline",
                run_index=0,
                judge_result=_make_judge_result(
                    unverified_claims=["claim A", "claim B"]
                ),
            ),
            _make_run(
                run_id="r",
                sample=s0,
                condition="baseline",
                run_index=1,
                judge_result=_make_judge_result(
                    unverified_claims=["claim B", "claim C"]
                ),
            ),
        ]

        results = aggregate(runs=runs)
        result = results[0]

        assert set(result.unverified_claims) == {"claim A", "claim B", "claim C"}

    def test_no_duplicate_claims(self) -> None:
        s0 = _make_sample(idx="0")
        runs = [
            _make_run(
                run_id="r",
                sample=s0,
                condition="baseline",
                run_index=0,
                judge_result=_make_judge_result(unverified_claims=["same claim"]),
            ),
            _make_run(
                run_id="r",
                sample=s0,
                condition="baseline",
                run_index=1,
                judge_result=_make_judge_result(unverified_claims=["same claim"]),
            ),
        ]

        results = aggregate(runs=runs)
        result = results[0]

        assert result.unverified_claims.count("same claim") == 1

    def test_empty_claims_when_no_runs_have_claims(self) -> None:
        s0 = _make_sample(idx="0")
        runs = [
            _make_run(
                run_id="r",
                sample=s0,
                condition="baseline",
                run_index=0,
                judge_result=_make_judge_result(unverified_claims=[]),
            ),
        ]

        results = aggregate(runs=runs)
        result = results[0]

        assert result.unverified_claims == []


class TestAggregateFields:
    """AggregatedResult carries the correct sample, condition, and runs."""

    def test_sample_is_preserved(self) -> None:
        s0 = _make_sample(idx="42")
        runs = [_make_run(run_id="r", sample=s0, condition="baseline", run_index=0)]

        results = aggregate(runs=runs)

        assert results[0].sample == s0

    def test_condition_is_preserved(self) -> None:
        s0 = _make_sample(idx="0")
        runs = [_make_run(run_id="r", sample=s0, condition="with-graph", run_index=0)]

        results = aggregate(runs=runs)

        assert results[0].condition == "with-graph"

    def test_conditions_are_keyed_separately(self) -> None:
        s0 = _make_sample(idx="0")
        runs = [
            _make_run(run_id="r", sample=s0, condition="baseline", run_index=0),
            _make_run(run_id="r", sample=s0, condition="with-graph", run_index=0),
        ]

        results = aggregate(runs=runs)
        conditions = {r.condition for r in results}

        assert conditions == {"baseline", "with-graph"}
