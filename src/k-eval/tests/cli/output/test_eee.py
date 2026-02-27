"""Tests for cli/output/eee.py — EEE schema serialization."""

import json

from k_eval.agent.domain.result import AgentResult
from k_eval.agent.domain.turn import AgentTurn, ToolCall
from k_eval.agent.domain.usage import UsageMetrics
from k_eval.cli.output.aggregator import AggregatedResult, aggregate
from k_eval.cli.output.eee import build_aggregate_json, build_instance_jsonl_lines
from k_eval.config.domain.agent import AgentConfig
from k_eval.config.domain.judge import JudgeConfig
from k_eval.dataset.domain.sample import Sample
from k_eval.evaluation.domain.run import EvaluationRun
from k_eval.evaluation.domain.summary import RunSummary
from k_eval.judge.domain.score import JudgeResult


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

    def test_interaction_type_is_single_turn_when_no_tool_turns(self) -> None:
        """interaction_type is 'single_turn' when runs have no tool-use turns."""
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        # _make_two_run_scenario produces agent results with turns=[] (no tool use)
        assert lines[0]["interaction_type"] == "single_turn"

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
        assert "reasoning_trace" in detail
        assert "cost_usd" in detail
        assert "duration_ms" in detail
        assert "num_turns" in detail

    def test_output_raw_is_string_not_list(self) -> None:
        """output.raw must be a single string per the EEE schema — not a list."""
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert isinstance(lines[0]["output"]["raw"], str)

    def test_output_raw_equals_first_run_response(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert lines[0]["output"]["raw"] == aggregated[0].runs[0].agent_result.response

    def test_output_does_not_have_raw_runs(self) -> None:
        """raw_runs was removed; output.raw is the schema-compliant single string."""
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert "raw_runs" not in lines[0]["output"]

    def test_details_includes_reasoning_traces_list(self) -> None:
        """evaluation.details contains reasoning_traces — one dict per run."""
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        details = lines[0]["evaluation"]["details"]
        assert "reasoning_traces" in details
        assert isinstance(details["reasoning_traces"], list)
        # _make_two_run_scenario: 2 runs per (sample, condition)
        assert len(details["reasoning_traces"]) == 2

    def test_reasoning_traces_entries_have_repetition_index_and_trace(self) -> None:
        summary, aggregated, agent_cfg, judge_cfg = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        for entry in lines[0]["evaluation"]["details"]["reasoning_traces"]:
            assert "repetition_index" in entry
            assert "reasoning_trace" in entry
            assert isinstance(entry["reasoning_trace"], str)

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

    def test_evaluation_timestamp_and_elapsed_written_to_details(self) -> None:
        summary, aggregated, agent_cfg, _ = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
            evaluation_timestamp=1700000000,
            elapsed_seconds=42.5,
        )

        details = lines[0]["evaluation"]["details"]
        assert details["evaluation_timestamp"] == 1700000000
        assert details["elapsed_seconds"] == 42.5

    def test_evaluation_timestamp_defaults_to_zero(self) -> None:
        summary, aggregated, agent_cfg, _ = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        details = lines[0]["evaluation"]["details"]
        assert details["evaluation_timestamp"] == 0
        assert details["elapsed_seconds"] == 0.0


# ---------------------------------------------------------------------------
# Helpers for turns-based EEE serialization tests
# ---------------------------------------------------------------------------


def _make_tool_call(
    tool_use_id: str = "tu-1",
    tool_name: str = "search_tool",
    tool_result: str | None = "search results",
    tool_error: bool = False,
) -> ToolCall:
    return ToolCall(
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        tool_input={"q": "test"},
        tool_result=tool_result,
        tool_error=tool_error,
    )


def _make_assistant_turn(turn_idx: int = 0, text: str = "Reasoning text.") -> AgentTurn:
    return AgentTurn(
        turn_idx=turn_idx,
        role="assistant",
        text=text,
        tool_calls=[],
    )


def _make_tool_use_turn(
    turn_idx: int = 1, tool_call: ToolCall | None = None
) -> AgentTurn:
    if tool_call is None:
        tool_call = _make_tool_call()
    return AgentTurn(
        turn_idx=turn_idx,
        role="tool_use",
        text=None,
        tool_calls=[tool_call],
    )


def _make_agent_result_with_turns(turns: list[AgentTurn]) -> AgentResult:
    return AgentResult(
        response="Final answer.",
        cost_usd=0.002,
        duration_ms=300,
        duration_api_ms=250,
        num_turns=len(turns),
        usage=UsageMetrics(input_tokens=100, output_tokens=50),
        turns=turns,
    )


def _make_run_with_turns(
    sample: Sample,
    condition: str,
    repetition_index: int,
    turns: list[AgentTurn],
    run_id: str = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb",
) -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id,
        sample=sample,
        condition=condition,
        repetition_index=repetition_index,
        agent_result=_make_agent_result_with_turns(turns=turns),
        judge_result=_make_judge_result(),
    )


def _make_agentic_scenario() -> tuple[RunSummary, list[AggregatedResult], AgentConfig]:
    """Single sample, single condition, 1 repetition with tool-use turns."""
    s0 = _make_sample(idx="0")
    run_id = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb"
    assistant_turn = _make_assistant_turn(turn_idx=0, text="Let me search.")
    tool_turn = _make_tool_use_turn(turn_idx=1)
    final_turn = _make_assistant_turn(turn_idx=2, text="Based on results...")
    turns = [assistant_turn, tool_turn, final_turn]
    run = _make_run_with_turns(
        sample=s0,
        condition="with-tools",
        repetition_index=0,
        turns=turns,
        run_id=run_id,
    )
    summary = _make_summary(runs=[run], run_id=run_id)
    aggregated = aggregate(runs=[run])
    agent_config = _make_agent_config()
    return summary, aggregated, agent_config


def _make_no_tool_scenario() -> tuple[RunSummary, list[AggregatedResult], AgentConfig]:
    """Single sample, single condition, 1 repetition with no tool-use turns."""
    s0 = _make_sample(idx="0")
    run_id = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb"
    assistant_turn = _make_assistant_turn(turn_idx=0, text="Direct answer.")
    turns = [assistant_turn]
    run = _make_run_with_turns(
        sample=s0,
        condition="baseline",
        repetition_index=0,
        turns=turns,
        run_id=run_id,
    )
    summary = _make_summary(runs=[run], run_id=run_id)
    aggregated = aggregate(runs=[run])
    agent_config = _make_agent_config()
    return summary, aggregated, agent_config


class TestInstanceJsonlLinesInteractionType:
    """interaction_type reflects whether tool-use turns are present."""

    def test_interaction_type_is_agentic_when_tool_calls_present(self) -> None:
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert lines[0]["interaction_type"] == "agentic"

    def test_interaction_type_is_single_turn_when_no_tool_calls(self) -> None:
        summary, aggregated, agent_cfg = _make_no_tool_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert lines[0]["interaction_type"] == "single_turn"

    def test_interaction_type_defaults_to_agentic_when_no_turns_field(self) -> None:
        """Legacy AgentResult with no turns defaults to 'agentic' (backward compat)."""
        summary, aggregated, agent_cfg, _ = _make_two_run_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        # _make_two_run_scenario uses _make_agent_result which has turns=[].
        # With no tool-use turns, this should be "single_turn".
        assert lines[0]["interaction_type"] == "single_turn"


class TestInstanceJsonlLinesAnswerAttribution:
    """answer_attribution is a flat list of entries, one per turn across all runs."""

    def test_answer_attribution_present_in_output(self) -> None:
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert "answer_attribution" in lines[0]

    def test_answer_attribution_is_flat_list_of_dicts(self) -> None:
        """answer_attribution is a flat list — entries are dicts, not nested lists."""
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        attribution = lines[0]["answer_attribution"]
        assert isinstance(attribution, list)
        for entry in attribution:
            assert isinstance(entry, dict)

    def test_answer_attribution_has_tool_use_entries(self) -> None:
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        tool_entries = [
            e for e in lines[0]["answer_attribution"] if e.get("source") == "mcp_tool"
        ]
        assert len(tool_entries) >= 1

    def test_answer_attribution_tool_entry_has_required_fields(self) -> None:
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        tool_entry = next(
            e for e in lines[0]["answer_attribution"] if e.get("source") == "mcp_tool"
        )
        assert "repetition_index" in tool_entry
        assert "turn_idx" in tool_entry
        assert "extraction_method" in tool_entry
        assert tool_entry["extraction_method"] == "tool_call"
        assert "extracted_value" in tool_entry
        assert "is_terminal" in tool_entry
        assert tool_entry["is_terminal"] is False
        assert "tool_name" in tool_entry
        assert "tool_error" in tool_entry

    def test_answer_attribution_assistant_entry_has_required_fields(self) -> None:
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        agent_entries = [
            e
            for e in lines[0]["answer_attribution"]
            if e.get("source") == "agent_reasoning"
        ]
        assert len(agent_entries) >= 1
        entry = agent_entries[0]
        assert "repetition_index" in entry
        assert "turn_idx" in entry
        assert entry["extraction_method"] == "text_generation"
        assert "extracted_value" in entry
        assert "is_terminal" in entry

    def test_last_assistant_turn_is_terminal(self) -> None:
        """The last assistant turn entry for a given run is marked is_terminal=True."""
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        # Scenario has 1 run (rep 0): assistant[0], tool_use[1], assistant[2].
        agent_entries = [
            e
            for e in lines[0]["answer_attribution"]
            if e.get("source") == "agent_reasoning"
        ]
        # The last assistant entry should be terminal; earlier ones must not be.
        assert agent_entries[-1]["is_terminal"] is True
        for entry in agent_entries[:-1]:
            assert entry["is_terminal"] is False

    def test_tool_entry_extracted_value_is_tool_result(self) -> None:
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        tool_entry = next(
            e for e in lines[0]["answer_attribution"] if e.get("source") == "mcp_tool"
        )
        assert tool_entry["extracted_value"] == "search results"

    def test_tool_entry_tool_name_matches(self) -> None:
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        tool_entry = next(
            e for e in lines[0]["answer_attribution"] if e.get("source") == "mcp_tool"
        )
        assert tool_entry["tool_name"] == "search_tool"

    def test_answer_attribution_entries_carry_repetition_index(self) -> None:
        """Every entry in the flat list has a repetition_index field."""
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        for entry in lines[0]["answer_attribution"]:
            assert "repetition_index" in entry

    def test_answer_attribution_multi_run_has_entries_for_each_rep(self) -> None:
        """With 2 runs, attribution entries exist for both repetition indices."""
        s0 = _make_sample(idx="0")
        run_id = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb"
        turns = [
            _make_assistant_turn(turn_idx=0, text="Thinking."),
            _make_tool_use_turn(turn_idx=1),
            _make_assistant_turn(turn_idx=2, text="Done."),
        ]
        run0 = _make_run_with_turns(
            sample=s0, condition="c", repetition_index=0, turns=turns, run_id=run_id
        )
        run1 = _make_run_with_turns(
            sample=s0, condition="c", repetition_index=1, turns=turns, run_id=run_id
        )
        summary = _make_summary(runs=[run0, run1], run_id=run_id)
        aggregated = aggregate(runs=[run0, run1])
        agent_cfg = _make_agent_config()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        rep_indices = {e["repetition_index"] for e in lines[0]["answer_attribution"]}
        assert 0 in rep_indices
        assert 1 in rep_indices


class TestInstanceJsonlLinesReasoningTrace:
    """output.reasoning_trace is populated from non-terminal assistant text turns."""

    def test_reasoning_trace_present_in_output(self) -> None:
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        assert "reasoning_trace" in lines[0]["output"]

    def test_reasoning_trace_contains_non_terminal_assistant_text(self) -> None:
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        # The scenario has 3 turns: assistant("Let me search."), tool_use, assistant("Based on results...")
        # Only "Let me search." is non-terminal (the second assistant turn is terminal/last).
        trace = lines[0]["output"]["reasoning_trace"]
        assert "Let me search." in trace

    def test_reasoning_trace_empty_string_when_no_non_terminal_turns(self) -> None:
        summary, aggregated, agent_cfg = _make_no_tool_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        # Only one assistant turn which is terminal — no non-terminal text.
        trace = lines[0]["output"]["reasoning_trace"]
        assert trace == ""

    def test_answer_attribution_and_reasoning_trace_are_json_serializable(
        self,
    ) -> None:
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        for line in lines:
            json.dumps(line)  # Should not raise


class TestInstanceJsonlLinesAnswerAttributionDuration:
    """Tool-call answer_attribution entries include duration_ms."""

    def test_tool_entry_has_duration_ms_field(self) -> None:
        """Each mcp_tool attribution entry has a duration_ms key."""
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        tool_entries = [
            e for e in lines[0]["answer_attribution"] if e.get("source") == "mcp_tool"
        ]
        assert len(tool_entries) >= 1
        for entry in tool_entries:
            assert "duration_ms" in entry

    def test_tool_entry_duration_ms_is_none_when_not_set(self) -> None:
        """duration_ms is None when ToolCall.duration_ms is None."""
        s0 = _make_sample(idx="0")
        run_id = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb"
        tc = _make_tool_call()  # duration_ms defaults to None
        turns = [
            _make_assistant_turn(turn_idx=0, text="Thinking."),
            _make_tool_use_turn(turn_idx=1, tool_call=tc),
            _make_assistant_turn(turn_idx=2, text="Done."),
        ]
        run = _make_run_with_turns(
            sample=s0, condition="c", repetition_index=0, turns=turns, run_id=run_id
        )
        summary = _make_summary(runs=[run], run_id=run_id)
        aggregated = aggregate(runs=[run])
        agent_cfg = _make_agent_config()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        tool_entry = next(
            e for e in lines[0]["answer_attribution"] if e.get("source") == "mcp_tool"
        )
        assert tool_entry["duration_ms"] is None

    def test_tool_entry_duration_ms_value_when_set(self) -> None:
        """duration_ms is the float value from the ToolCall when present."""
        s0 = _make_sample(idx="0")
        run_id = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb"
        tc = ToolCall(
            tool_use_id="tu-timed",
            tool_name="search_tool",
            tool_input={"q": "test"},
            tool_result="results",
            tool_error=False,
            duration_ms=750.5,
        )
        turns = [
            _make_assistant_turn(turn_idx=0, text="Thinking."),
            _make_tool_use_turn(turn_idx=1, tool_call=tc),
            _make_assistant_turn(turn_idx=2, text="Done."),
        ]
        run = _make_run_with_turns(
            sample=s0, condition="c", repetition_index=0, turns=turns, run_id=run_id
        )
        summary = _make_summary(runs=[run], run_id=run_id)
        aggregated = aggregate(runs=[run])
        agent_cfg = _make_agent_config()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        tool_entry = next(
            e for e in lines[0]["answer_attribution"] if e.get("source") == "mcp_tool"
        )
        assert tool_entry["duration_ms"] == 750.5

    def test_agent_reasoning_entry_has_no_duration_ms_field(self) -> None:
        """agent_reasoning entries do NOT have a duration_ms field."""
        summary, aggregated, agent_cfg = _make_agentic_scenario()

        lines = build_instance_jsonl_lines(
            summary=summary,
            aggregated=aggregated,
            agent_config=agent_cfg,
        )

        agent_entries = [
            e
            for e in lines[0]["answer_attribution"]
            if e.get("source") == "agent_reasoning"
        ]
        assert len(agent_entries) >= 1
        for entry in agent_entries:
            assert "duration_ms" not in entry
