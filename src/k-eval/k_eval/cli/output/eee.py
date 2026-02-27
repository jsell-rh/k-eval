"""EEE schema v0.2.1 serialization — aggregate JSON and instance JSONL."""

import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from k_eval.agent.domain.turn import AgentTurn
from k_eval.cli.output.aggregator import AggregatedResult
from k_eval.config.domain.agent import AgentConfig
from k_eval.config.domain.judge import JudgeConfig
from k_eval.evaluation.domain.run import EvaluationRun
from k_eval.evaluation.domain.summary import RunSummary

type JsonDict = dict[str, Any]


def _k_eval_version() -> str:
    try:
        return version("k-eval")
    except PackageNotFoundError:
        return "dev"


def _source_metadata() -> JsonDict:
    return {
        "source_name": "k-eval",
        "source_type": "evaluation_run",
        "source_organization_name": "k-eval",
        "evaluator_relationship": "self",
    }


def _metric_config() -> JsonDict:
    return {
        "evaluation_description": (
            "k-eval LLM-as-judge scoring: factual_adherence, completeness, "
            "helpfulness_and_clarity (each 1-5)"
        ),
        "lower_is_better": False,
        "score_type": "composite",
        "min_score": 1.0,
        "max_score": 5.0,
    }


def _aggregate_score_details_for_condition(
    condition: str,
    aggregated: list[AggregatedResult],
) -> JsonDict:
    """Compute per-metric mean ± stddev across all samples for one condition."""
    condition_results = [r for r in aggregated if r.condition == condition]

    if not condition_results:
        return {}

    # Average the per-sample means and stddevs across all samples for this condition.
    fa_means = [r.factual_adherence_mean for r in condition_results]
    fa_stds = [r.factual_adherence_stddev for r in condition_results]
    co_means = [r.completeness_mean for r in condition_results]
    co_stds = [r.completeness_stddev for r in condition_results]
    hc_means = [r.helpfulness_and_clarity_mean for r in condition_results]
    hc_stds = [r.helpfulness_and_clarity_stddev for r in condition_results]

    n = len(condition_results)
    return {
        "factual_adherence_mean": sum(fa_means) / n,
        "factual_adherence_stddev": sum(fa_stds) / n,
        "completeness_mean": sum(co_means) / n,
        "completeness_stddev": sum(co_stds) / n,
        "helpfulness_and_clarity_mean": sum(hc_means) / n,
        "helpfulness_and_clarity_stddev": sum(hc_stds) / n,
    }


def build_aggregate_json(
    summary: RunSummary,
    aggregated: list[AggregatedResult],
    agent_config: AgentConfig,
    judge_config: JudgeConfig,
) -> JsonDict:
    """Build the EEE aggregate JSON dict (one entry per condition).

    Returns a plain dict that is JSON-serializable.
    """
    now_ts = str(int(time.time()))
    conditions = sorted({r.condition for r in aggregated})
    total_samples = len({r.sample.sample_idx for r in aggregated})

    evaluation_results: list[JsonDict] = []
    for condition in conditions:
        score_details = _aggregate_score_details_for_condition(
            condition=condition,
            aggregated=aggregated,
        )
        evaluation_results.append(
            {
                "evaluation_name": f"{summary.config_name}/{condition}",
                "evaluation_timestamp": now_ts,
                "source_data": {
                    "source_type": "jsonl_file",
                    "dataset_sha256": summary.dataset_sha256,
                    "samples_number": total_samples,
                },
                "metric_config": _metric_config(),
                "score_details": {
                    "score": None,
                    "details": score_details,
                },
                "generation_config": {
                    "judge_temperature": judge_config.temperature,
                },
            }
        )

    return {
        "schema_version": "0.2.1",
        "evaluation_id": summary.run_id,
        "retrieved_timestamp": now_ts,
        "source_metadata": _source_metadata(),
        "model_info": {
            "id": agent_config.model,
            "name": agent_config.model,
            "developer": None,
            "inference_platform": None,
        },
        "eval_library": {
            "name": "k-eval",
            "version": _k_eval_version(),
        },
        "evaluation_results": evaluation_results,
        "detailed_evaluation_results": {
            "file_path": None,  # Set by caller after computing stem
            "format": "jsonl",
            "checksum": summary.dataset_sha256,
        },
    }


def _sum_tokens(
    aggregated_result: AggregatedResult,
    token_attr: str,
) -> int | None:
    """Sum a token attribute across all runs; return None if any run lacks it."""
    total = 0
    for run in aggregated_result.runs:
        usage = run.agent_result.usage
        if usage is None:
            return None
        val = getattr(usage, token_attr)
        if val is None:
            return None
        total += val
    return total


def _has_tool_use_turns(turns: list[AgentTurn]) -> bool:
    """Return True if any turn is a tool_use turn."""
    return any(t.role == "tool_use" for t in turns)


def _build_run_answer_attribution(
    run: EvaluationRun,
) -> list[JsonDict]:
    """Build attribution entries for a single run.

    Each tool_use turn produces one entry per ToolCall (source="mcp_tool").
    Each assistant turn produces one entry (source="agent_reasoning").
    The last assistant turn is marked is_terminal=True.
    Each entry carries repetition_index so callers can identify the run.
    """
    turns = run.agent_result.turns
    # Identify the list-index of the last assistant turn for is_terminal logic.
    last_assistant_idx: int | None = None
    for i, turn in enumerate(turns):
        if turn.role == "assistant":
            last_assistant_idx = i

    entries: list[JsonDict] = []
    for i, turn in enumerate(turns):
        if turn.role == "tool_use":
            for tc in turn.tool_calls:
                entries.append(
                    {
                        "repetition_index": run.repetition_index,
                        "turn_idx": turn.turn_idx,
                        "source": "mcp_tool",
                        "extracted_value": tc.tool_result
                        if tc.tool_result is not None
                        else "",
                        "extraction_method": "tool_call",
                        "is_terminal": False,
                        "tool_name": tc.tool_name,
                        "tool_error": tc.tool_error,
                        "duration_ms": tc.duration_ms,
                    }
                )
        elif turn.role == "assistant":
            is_terminal = i == last_assistant_idx
            entries.append(
                {
                    "repetition_index": run.repetition_index,
                    "turn_idx": turn.turn_idx,
                    "source": "agent_reasoning",
                    "extracted_value": turn.text or "",
                    "extraction_method": "text_generation",
                    "is_terminal": is_terminal,
                }
            )
    return entries


def _build_reasoning_trace(run: EvaluationRun) -> str:
    """Concatenate text from non-terminal assistant turns for reasoning_trace."""
    turns = run.agent_result.turns
    # Identify the last assistant turn index.
    last_assistant_idx: int | None = None
    for i, turn in enumerate(turns):
        if turn.role == "assistant":
            last_assistant_idx = i

    parts: list[str] = []
    for i, turn in enumerate(turns):
        if turn.role == "assistant" and i != last_assistant_idx:
            if turn.text:
                parts.append(turn.text)
    return " ".join(parts)


def build_instance_jsonl_lines(
    summary: RunSummary,
    aggregated: list[AggregatedResult],
    agent_config: AgentConfig,
    evaluation_timestamp: int = 0,
    elapsed_seconds: float = 0.0,
) -> list[JsonDict]:
    """Build one JSONL line dict per AggregatedResult.

    Returns a list of plain dicts, each JSON-serializable.
    """
    lines: list[JsonDict] = []

    for agg in aggregated:
        fa_reasonings = [r.judge_result.factual_adherence_reasoning for r in agg.runs]
        co_reasonings = [r.judge_result.completeness_reasoning for r in agg.runs]
        hc_reasonings = [
            r.judge_result.helpfulness_and_clarity_reasoning for r in agg.runs
        ]

        # Fix 1 — run_details: add reasoning_trace per entry.
        run_details: list[JsonDict] = [
            {
                "repetition_index": run.repetition_index,
                "agent_response": run.agent_result.response,
                "reasoning_trace": _build_reasoning_trace(run=run),
                "cost_usd": run.agent_result.cost_usd,
                "duration_ms": run.agent_result.duration_ms,
                "num_turns": run.agent_result.num_turns,
                "factual_adherence": run.judge_result.factual_adherence,
                "completeness": run.judge_result.completeness,
                "helpfulness_and_clarity": run.judge_result.helpfulness_and_clarity,
            }
            for run in agg.runs
        ]

        input_tokens = _sum_tokens(aggregated_result=agg, token_attr="input_tokens")
        output_tokens = _sum_tokens(aggregated_result=agg, token_attr="output_tokens")

        # interaction_type: "agentic" if any run has tool_use turns, else "single_turn"
        has_tool_use = any(
            _has_tool_use_turns(turns=run.agent_result.turns) for run in agg.runs
        )
        interaction_type = "agentic" if has_tool_use else "single_turn"

        # Fix 2 — answer_attribution: flat list with repetition_index on each entry.
        answer_attribution: list[JsonDict] = []
        for run in agg.runs:
            for entry in _build_run_answer_attribution(run=run):
                answer_attribution.append(entry)

        # Fix 3 — output.raw: single string (rep-0 response); reasoning_trace from rep-0.
        primary_response = agg.runs[0].agent_result.response if agg.runs else ""
        reasoning_trace = _build_reasoning_trace(run=agg.runs[0]) if agg.runs else ""

        # Fix 4 — evaluation.details: add reasoning_traces list (one per run).
        reasoning_traces: list[JsonDict] = [
            {
                "repetition_index": run.repetition_index,
                "reasoning_trace": _build_reasoning_trace(run=run),
            }
            for run in agg.runs
        ]

        lines.append(
            {
                "schema_version": "instance_level_eval_0.2.1",
                "evaluation_id": summary.run_id,
                "model_id": agent_config.model,
                "evaluation_name": f"{summary.config_name}/{agg.condition}",
                "sample_idx": agg.sample.sample_idx,
                "interaction_type": interaction_type,
                "input": {
                    "raw": agg.sample.question,
                    "reference": agg.sample.answer,
                },
                "output": {
                    "raw": primary_response,
                    "reasoning_trace": reasoning_trace,
                },
                "answer_attribution": answer_attribution,
                "evaluation": {
                    "score": None,
                    "details": {
                        "evaluation_timestamp": evaluation_timestamp,
                        "elapsed_seconds": round(elapsed_seconds, 1),
                        "factual_adherence_mean": agg.factual_adherence_mean,
                        "factual_adherence_stddev": agg.factual_adherence_stddev,
                        "factual_adherence_reasonings": fa_reasonings,
                        "completeness_mean": agg.completeness_mean,
                        "completeness_stddev": agg.completeness_stddev,
                        "completeness_reasonings": co_reasonings,
                        "helpfulness_and_clarity_mean": agg.helpfulness_and_clarity_mean,
                        "helpfulness_and_clarity_stddev": agg.helpfulness_and_clarity_stddev,
                        "helpfulness_and_clarity_reasonings": hc_reasonings,
                        "unverified_claims": agg.unverified_claims,
                        "reasoning_traces": reasoning_traces,
                    },
                },
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "run_details": run_details,
            }
        )

    return lines
