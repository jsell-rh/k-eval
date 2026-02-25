"""EEE schema v0.2.1 serialization — aggregate JSON and instance JSONL."""

import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from cli.output.aggregator import AggregatedResult
from config.domain.agent import AgentConfig
from config.domain.judge import JudgeConfig
from evaluation.domain.summary import RunSummary

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


def build_instance_jsonl_lines(
    summary: RunSummary,
    aggregated: list[AggregatedResult],
    agent_config: AgentConfig,
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

        run_details: list[JsonDict] = [
            {
                "repetition_index": run.repetition_index,
                "agent_response": run.agent_result.response,
                "cost_usd": run.agent_result.cost_usd,
                "duration_ms": run.agent_result.duration_ms,
                "num_turns": run.agent_result.num_turns,
            }
            for run in agg.runs
        ]

        input_tokens = _sum_tokens(aggregated_result=agg, token_attr="input_tokens")
        output_tokens = _sum_tokens(aggregated_result=agg, token_attr="output_tokens")

        lines.append(
            {
                "schema_version": "instance_level_eval_0.2.1",
                "evaluation_id": summary.run_id,
                "model_id": agent_config.model,
                "evaluation_name": f"{summary.config_name}/{agg.condition}",
                "sample_idx": agg.sample.sample_idx,
                "interaction_type": "agentic",
                "input": {
                    "raw": agg.sample.question,
                    "reference": agg.sample.answer,
                },
                "output": {
                    "raw_runs": [run.agent_result.response for run in agg.runs],
                },
                "evaluation": {
                    "score": None,
                    "details": {
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
