"""Tests for cli/output/eee.py — EEE schema serialization."""

import json

from agent.domain.result import AgentResult
from agent.domain.usage import UsageMetrics
from cli.output.aggregator import AggregatedResult, aggregate
from cli.output.eee import build_aggregate_json, build_instance_jsonl_lines
from config.domain.agent import AgentConfig
from config.domain.judge import JudgeConfig
from dataset.domain.sample import Sample
from evaluation.domain.run import EvaluationRun
from evaluation.domain.summary import RunSummary
from judge.domain.score import JudgeResult


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _make_sample(idx: str = "0") -> Sample:
    return Sample(
        sample_idx=idx,
        question=f"Question {idx}?",
        answer=f"Answer {idx}.",
    )


def _make_agent_result(response: str = "Agent answer.") -> AgentResult:
    return AgentResult(
        response=response,
        cost_usd=0.002,
        duration_ms=300,
        duration_api_ms=250,
        num_turns=2,
        usage=UsageMetrics(input_tokens=100, output_tokens=50),
    )


def _make_judge_result(
    factual_adherence: int = 4,
    completeness: int = 5,
    helpfulness_and_clarity: int = 3,
    unverified_claims: list[str] | None = None,
) -> JudgeResult:
    return JudgeResult(
        factual_adherence=factual_adherence,
        factual_adherence_reasoning="FA reasoning.",
        completeness=completeness,
        completeness_reasoning="CO reasoning.",
        helpfulness_and_clarity=helpfulness_and_clarity,
        helpfulness_and_clarity_reasoning="HC reasoning.",
        unverified_claims=unverified_claims or [],
    )


def _make_run(
    sample: Sample,
    condition: str,
    repetition_index: int,
    run_id: str = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb",
) -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id,
        sample=sample,
        condition=condition,
        repetition_index=repetition_index,
        agent_result=_make_agent_result(),
        judge_result=_make_judge_result(),
    )


def _make_summary(
    runs: list[EvaluationRun],
    run_id: str = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb",
    config_name: str = "test-eval",
    dataset_sha256: str = "abcdef1234567890" * 4,
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        dataset_sha256=dataset_sha256,
        config_name=config_name,
        runs=runs,
    )


def _make_agent_config(model: str = "claude-3-5-sonnet") -> AgentConfig:
    return AgentConfig(type="claude_code_sdk", model=model)


def _make_judge_config(temperature: float = 0.0) -> JudgeConfig:
    return JudgeConfig(model="gpt-4o", temperature=temperature)


def _make_two_run_scenario() -> tuple[
    RunSummary, list[AggregatedResult], AgentConfig, JudgeConfig
]:
    """2 samples × 1 condition × 2 runs → 2 AggregatedResults."""
    s0 = _make_sample(idx="0")
    s1 = _make_sample(idx="1")
    run_id = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb"
    runs = [
        _make_run(sample=s0, condition="baseline", repetition_index=0, run_id=run_id),
        _make_run(sample=s0, condition="baseline", repetition_index=1, run_id=run_id),
        _make_run(sample=s1, condition="baseline", repetition_index=0, run_id=run_id),
        _make_run(sample=s1, condition="baseline", repetition_index=1, run_id=run_id),
    ]
    summary = _make_summary(runs=runs, run_id=run_id)
    aggregated = aggregate(runs=runs)
    agent_config = _make_agent_config()
    judge_config = _make_judge_config()
    return summary, aggregated, agent_config, judge_config


# ---------------------------------------------------------------------------
# Tests for aggregate JSON
# ---------------------------------------------------------------------------


class TestAggregateJson:
    """build_aggregate_json returns a valid EEE aggregate JSON dict."""

    def test_schema_version_is_correct(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        assert result["schema_version"] == "0.2.1"

    def test_evaluation_id_matches_run_id(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        assert result["evaluation_id"] == summary.run_id

    def test_model_info_id_matches_agent_model(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        assert result["model_info"]["id"] == agent_cfg.model

    def test_model_info_developer_is_null(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        assert result["model_info"]["developer"] is None

    def test_source_metadata_source_name_is_k_eval(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        assert result["source_metadata"]["source_name"] == "k-eval"

    def test_evaluation_results_has_one_entry_per_condition(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        # 1 condition ("baseline"), 2 samples → 1 evaluation_result entry
        assert len(result["evaluation_results"]) == 1

    def test_evaluation_result_name_uses_config_slash_condition(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        assert (
            result["evaluation_results"][0]["evaluation_name"] == "test-eval/baseline"
        )

    def test_score_is_null(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        assert result["evaluation_results"][0]["score_details"]["score"] is None

    def test_details_has_all_six_metric_stat_fields(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        details = result["evaluation_results"][0]["score_details"]["details"]
        assert "factual_adherence_mean" in details
        assert "factual_adherence_stddev" in details
        assert "completeness_mean" in details
        assert "completeness_stddev" in details
        assert "helpfulness_and_clarity_mean" in details
        assert "helpfulness_and_clarity_stddev" in details

    def test_dataset_sha256_in_source_data(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        assert (
            result["evaluation_results"][0]["source_data"]["dataset_sha256"]
            == summary.dataset_sha256
        )

    def test_judge_temperature_in_generation_config(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        assert (
            result["evaluation_results"][0]["generation_config"]["judge_temperature"]
            == judge_cfg.temperature
        )

    def test_result_is_json_serializable(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        result = build_aggregate_json(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            judge_config=judge_cfg,
        )

        # Should not raise
        json.dumps(result)


# ---------------------------------------------------------------------------
# Tests for instance JSONL lines
# ---------------------------------------------------------------------------


class TestInstanceJsonlLines:
    """build_instance_jsonl_lines returns one dict per AggregatedResult."""

    def test_returns_one_line_per_aggregated_result(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert len(lines) == len(aggregated)

    def test_schema_version_is_instance_level(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert lines[0]["schema_version"] == "instance_level_eval_0.2.1"

    def test_interaction_type_is_agentic(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert lines[0]["interaction_type"] == "agentic"

    def test_score_is_null(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert lines[0]["evaluation"]["score"] is None

    def test_details_has_all_six_metric_stat_fields(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        details = lines[0]["evaluation"]["details"]
        assert "factual_adherence_mean" in details
        assert "factual_adherence_stddev" in details
        assert "completeness_mean" in details
        assert "completeness_stddev" in details
        assert "helpfulness_and_clarity_mean" in details
        assert "helpfulness_and_clarity_stddev" in details

    def test_details_includes_reasoning_lists(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        details = lines[0]["evaluation"]["details"]
        assert "factual_adherence_reasonings" in details
        assert "completeness_reasonings" in details
        assert "helpfulness_and_clarity_reasonings" in details

    def test_details_includes_unverified_claims(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert "unverified_claims" in lines[0]["evaluation"]["details"]

    def test_model_id_matches_agent_config(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert lines[0]["model_id"] == agent_cfg.model

    def test_evaluation_id_matches_run_id(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert lines[0]["evaluation_id"] == summary.run_id

    def test_sample_idx_matches_sample(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert lines[0]["sample_idx"] == aggregated[0].sample.sample_idx

    def test_input_raw_is_sample_question(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert lines[0]["input"]["raw"] == aggregated[0].sample.question

    def test_input_reference_is_sample_answer(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert lines[0]["input"]["reference"] == aggregated[0].sample.answer

    def test_run_details_has_one_entry_per_run(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        # 2 runs per (sample, condition)
        assert len(lines[0]["run_details"]) == 2

    def test_run_details_contains_expected_fields(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        detail = lines[0]["run_details"][0]
        assert "repetition_index" in detail
        assert "agent_response" in detail
        assert "cost_usd" in detail
        assert "duration_ms" in detail
        assert "num_turns" in detail

    def test_each_line_is_json_serializable(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        for line in lines:
            json.dumps(line)  # Should not raise

    def test_token_usage_sums_across_runs(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        # Each run has 100 input and 50 output tokens, 2 runs per group
        assert lines[0]["token_usage"]["input_tokens"] == 200
        assert lines[0]["token_usage"]["output_tokens"] == 100
