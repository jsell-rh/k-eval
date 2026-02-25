"""Aggregator — groups EvaluationRuns by (sample, condition) and computes statistics."""

import statistics
from dataclasses import dataclass

from k_eval.dataset.domain.sample import Sample
from k_eval.evaluation.domain.run import EvaluationRun

type GroupKey = tuple[str, str]  # (sample_idx, condition)


@dataclass(frozen=True)
class AggregatedResult:
    """One (sample, condition) pair with aggregated scores across N runs."""

    sample: Sample
    condition: str
    runs: list[EvaluationRun]

    factual_adherence_mean: float
    factual_adherence_stddev: float
    completeness_mean: float
    completeness_stddev: float
    helpfulness_and_clarity_mean: float
    helpfulness_and_clarity_stddev: float

    unverified_claims: list[str]


def _stddev(values: list[float]) -> float:
    """Return sample stddev for N >= 2, else 0.0."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def aggregate(runs: list[EvaluationRun]) -> list[AggregatedResult]:
    """Group EvaluationRuns by (sample_idx, condition) and compute per-metric statistics.

    Returns one AggregatedResult per unique (sample, condition) pair, preserving
    insertion order of first occurrence.
    """
    # Preserve insertion order via dict keyed by (sample_idx, condition).
    groups: dict[GroupKey, list[EvaluationRun]] = {}
    samples: dict[GroupKey, Sample] = {}

    for run in runs:
        key: GroupKey = (run.sample.sample_idx, run.condition)
        if key not in groups:
            groups[key] = []
            samples[key] = run.sample
        groups[key].append(run)

    results: list[AggregatedResult] = []
    for key, group_runs in groups.items():
        sample = samples[key]
        condition = key[1]

        sorted_runs = sorted(group_runs, key=lambda r: r.repetition_index)

        fa_scores = [float(r.judge_result.factual_adherence) for r in sorted_runs]
        co_scores = [float(r.judge_result.completeness) for r in sorted_runs]
        hc_scores = [float(r.judge_result.helpfulness_and_clarity) for r in sorted_runs]

        # Deduplicate unverified claims across all runs.
        seen: set[str] = set()
        deduped_claims: list[str] = []
        for run in sorted_runs:
            for claim in run.judge_result.unverified_claims:
                if claim not in seen:
                    seen.add(claim)
                    deduped_claims.append(claim)

        results.append(
            AggregatedResult(
                sample=sample,
                condition=condition,
                runs=sorted_runs,
                factual_adherence_mean=statistics.mean(fa_scores),
                factual_adherence_stddev=_stddev(fa_scores),
                completeness_mean=statistics.mean(co_scores),
                completeness_stddev=_stddev(co_scores),
                helpfulness_and_clarity_mean=statistics.mean(hc_scores),
                helpfulness_and_clarity_stddev=_stddev(hc_scores),
                unverified_claims=deduped_claims,
            )
        )

    return results
