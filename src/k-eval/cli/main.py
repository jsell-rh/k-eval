"""CLI entrypoint for k-eval — typer app with a `run` command."""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import structlog
import typer

from agent.infrastructure.observer import StructlogAgentObserver
from agent.infrastructure.registry import create_agent_factory
from cli.output.aggregator import AggregatedResult, aggregate
from cli.output.eee import build_aggregate_json, build_instance_jsonl_lines
from config.domain.agent import AgentConfig
from config.domain.judge import JudgeConfig
from config.infrastructure.observer import StructlogConfigObserver
from config.infrastructure.yaml_loader import YamlConfigLoader
from core.errors import KEvalError
from dataset.infrastructure.jsonl_loader import JsonlDatasetLoader
from dataset.infrastructure.observer import StructlogDatasetObserver
from evaluation.application.runner import EvaluationRunner
from evaluation.domain.observer import EvaluationObserver
from evaluation.domain.summary import RunSummary
from evaluation.infrastructure.composite_observer import CompositeEvaluationObserver
from evaluation.infrastructure.observer import StructlogEvaluationObserver
from evaluation.infrastructure.progress_observer import ProgressEvaluationObserver
from judge.infrastructure.factory import LiteLLMJudgeFactory
from judge.infrastructure.observer import StructlogJudgeObserver

app = typer.Typer(add_completion=False)


def _configure_structlog(log_format: str) -> None:
    """Configure structlog based on the requested format."""
    if log_format == "console":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    elif log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        typer.echo(f"Invalid log format: {log_format!r}. Must be 'console' or 'json'.")
        raise typer.Exit(code=1)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def _output_stem(config_name: str, run_id: str) -> str:
    """Build the output file stem: {config_name}_{YYYYMMDD}_{short_run_id}."""
    date_str = datetime.now().strftime("%Y%m%d")
    short_id = run_id[:8]
    return f"{config_name}_{date_str}_{short_id}"


def _write_outputs(
    output_dir: Path,
    stem: str,
    summary: RunSummary,
    aggregated: list[AggregatedResult],
    agent_config: AgentConfig,
    judge_config: JudgeConfig,
) -> tuple[Path, Path]:
    """Write EEE JSON and JSONL output files. Returns (json_path, jsonl_path)."""
    json_path = output_dir / f"{stem}.json"
    jsonl_path = output_dir / f"{stem}.detailed.jsonl"

    aggregate_data = build_aggregate_json(
        summary=summary,
        aggregated=aggregated,
        agent_config=agent_config,
        judge_config=judge_config,
    )
    # Patch in the relative JSONL filename.
    aggregate_data["detailed_evaluation_results"]["file_path"] = (
        f"{stem}.detailed.jsonl"
    )

    json_path.write_text(json.dumps(aggregate_data, indent=2), encoding="utf-8")

    instance_lines = build_instance_jsonl_lines(
        summary=summary,
        aggregated=aggregated,
        agent_config=agent_config,
    )
    jsonl_path.write_text(
        "\n".join(json.dumps(line) for line in instance_lines) + "\n",
        encoding="utf-8",
    )

    return json_path, jsonl_path


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_WHITE = "\033[97m"

# Maximum display width for a condition column header (chars, excluding padding).
_MAX_COND_LEN = 12
# Width of a single condition data cell: "X.XX±X.XX" = 9 chars, padded to 11.
_CELL_W = 11


def _score_color(score: float) -> str:
    if score >= 4.0:
        return _GREEN
    if score >= 3.0:
        return _YELLOW
    return _RED


def _rule(width: int = 72, color: str = _DIM) -> None:
    typer.echo(f"{color}{'─' * width}{_RESET}")


def _truncate(name: str, max_len: int = _MAX_COND_LEN) -> str:
    """Truncate a condition name to max_len, appending '…' if needed."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def _condition_mean(results: list[AggregatedResult], attr: str) -> float:
    """Compute the mean of a metric attribute across all (sample, condition) results."""
    n = len(results)
    return float(sum(float(getattr(r, f"{attr}_mean")) for r in results) / n)


def _condition_stddev(results: list[AggregatedResult], attr: str) -> float:
    """Compute the pooled (average) stddev of a metric across all results."""
    n = len(results)
    return float(sum(float(getattr(r, f"{attr}_stddev")) for r in results) / n)


def _overall_mean(results: list[AggregatedResult], metric_attrs: list[str]) -> float:
    """Return unweighted mean across all metrics for the given condition's results."""
    return sum(_condition_mean(results=results, attr=a) for a in metric_attrs) / len(
        metric_attrs
    )


def _cell(mean: float, std: float, *, is_winner: bool, multi_condition: bool) -> str:
    """
    Format a data cell: "X.XX±X.XX" (9 chars) with ANSI color.

    When multi_condition is False (single-condition view) we omit the ± suffix
    and show just the mean to keep the table narrow.
    """
    color = _score_color(score=mean)
    winner_marker = f"{_BOLD}▲{_RESET}" if is_winner else " "
    if multi_condition:
        cell_text = f"{mean:.2f}±{std:.2f}"
        # 9 chars of text, left-padded inside the fixed cell width
        return f"{color}{cell_text:>{_CELL_W - 1}}{_RESET}{winner_marker}"
    else:
        cell_text = f"{mean:.2f}"
        return f"{color}{cell_text:>{_CELL_W - 1}}{_RESET} "


def _print_single_condition(
    condition: str,
    results: list[AggregatedResult],
    metrics: list[tuple[str, str]],
    metric_w: int,
) -> None:
    """Render a simple single-condition table with mean, stddev, and a bar."""
    typer.echo("")
    _rule(color=_BLUE)
    typer.echo(f"{_BLUE}{_BOLD}  {condition}{_RESET}")
    _rule(color=_BLUE)
    typer.echo(
        f"  {_DIM}{'Metric':<{metric_w}}  {'Mean':>6}  {'±StdDev':>7}  Bar{_RESET}"
    )
    typer.echo(f"  {'─' * metric_w}  {'─' * 6}  {'─' * 7}  {'─' * 10}")

    for label, attr in metrics:
        mean = _condition_mean(results=results, attr=attr)
        std = _condition_stddev(results=results, attr=attr)
        color = _score_color(score=mean)
        filled = round(mean)
        bar = f"{color}{'█' * filled}{_DIM}{'░' * (5 - filled)}{_RESET}"
        typer.echo(
            f"  {_WHITE}{label:<{metric_w}}{_RESET}"
            f"  {color}{mean:>6.2f}{_RESET}"
            f"  {_DIM}{std:>7.2f}{_RESET}"
            f"  {bar}"
        )


def _print_comparison_table(
    conditions: list[str],
    results_by_condition: dict[str, list[AggregatedResult]],
    metrics: list[tuple[str, str]],
    metric_w: int,
) -> None:
    """
    Render a side-by-side comparison table: conditions as columns, metrics as rows.

    Each cell shows "mean±stddev". The winning value per row is highlighted in
    bold green with a ▲ marker. A final Δ column shows max-min spread.
    A summary row shows overall (cross-metric) mean per condition.
    """
    metric_attrs = [attr for _, attr in metrics]
    truncated = {c: _truncate(name=c) for c in conditions}

    # Header row — condition names as columns.
    header = f"  {_DIM}{'Metric':<{metric_w}}{_RESET}"
    for cond in conditions:
        label = truncated[cond]
        header += f"  {_CYAN}{_BOLD}{label:>{_CELL_W}}{_RESET}"
    typer.echo(header)

    # Separator.
    sep_len = metric_w + len(conditions) * (_CELL_W + 2)
    typer.echo(f"  {'─' * sep_len}")

    # One row per metric.
    for label, attr in metrics:
        means = {
            c: _condition_mean(results=results_by_condition[c], attr=attr)
            for c in conditions
        }
        stds = {
            c: _condition_stddev(results=results_by_condition[c], attr=attr)
            for c in conditions
        }
        max_mean = max(means.values())
        all_tied = (max(means.values()) - min(means.values())) < 0.005

        row = f"  {_WHITE}{label:<{metric_w}}{_RESET}"
        for cond in conditions:
            is_winner = (not all_tied) and (means[cond] >= max_mean - 0.0005)
            row += "  " + _cell(
                mean=means[cond],
                std=stds[cond],
                is_winner=is_winner,
                multi_condition=True,
            )
        typer.echo(row)

    # Overall summary row.
    typer.echo(f"  {'─' * sep_len}")
    overall = {
        c: _overall_mean(results=results_by_condition[c], metric_attrs=metric_attrs)
        for c in conditions
    }
    max_overall = max(overall.values())
    all_tied_overall = (max(overall.values()) - min(overall.values())) < 0.005

    overall_row = f"  {_DIM}{'Overall avg':<{metric_w}}{_RESET}"
    for cond in conditions:
        is_winner = (not all_tied_overall) and (overall[cond] >= max_overall - 0.0005)
        color = _score_color(score=overall[cond])
        winner_marker = f"{_BOLD}▲{_RESET}" if is_winner else " "
        cell_text = f"{overall[cond]:.2f}"
        overall_row += f"  {color}{cell_text:>{_CELL_W - 1}}{_RESET}{winner_marker}"
    typer.echo(overall_row)

    # Winner callout — only when there is a clear winner.
    if not all_tied_overall:
        winner_cond = max(conditions, key=lambda c: overall[c])
        winner_score = overall[winner_cond]
        runner_up_score = sorted(overall.values(), reverse=True)[1]
        advantage = winner_score - runner_up_score
        typer.echo("")
        typer.echo(
            f"  {_GREEN}{_BOLD}Winner: {winner_cond}{_RESET}"
            f"  {_DIM}overall avg {winner_score:.2f}"
            f"  (+{advantage:.2f} vs next){_RESET}"
        )

    # Unverified claims across all conditions.
    all_claims: list[tuple[str, str]] = []
    for cond in conditions:
        for r in results_by_condition[cond]:
            for claim in r.unverified_claims:
                all_claims.append((cond, claim))

    if all_claims:
        typer.echo("")
        typer.echo(
            f"  {_YELLOW}{_BOLD}Unverified claims  ({len(all_claims)} total){_RESET}"
        )
        for cond, claim in all_claims[:10]:
            short = claim[:60] + ("…" if len(claim) > 60 else "")
            typer.echo(f"  {_DIM}[{cond}]{_RESET} {short}")
        if len(all_claims) > 10:
            typer.echo(
                f"  {_DIM}… and {len(all_claims) - 10} more — see detailed JSONL{_RESET}"
            )


def _format_elapsed(elapsed_seconds: float) -> str:
    """Format elapsed seconds as '1m 23.4s' or '5.2s'."""
    minutes, seconds = divmod(elapsed_seconds, 60)
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:.1f}s"
    return f"{elapsed_seconds:.1f}s"


def _print_summary(
    summary: RunSummary,
    aggregated: list[AggregatedResult],
    json_path: Path,
    jsonl_path: Path,
    elapsed_seconds: float,
) -> None:
    """Print a colorized summary to stdout.

    For a single condition: simple table with mean, stddev, and bar chart.
    For multiple conditions: side-by-side comparison table with winner callout.
    """
    conditions = sorted({r.condition for r in aggregated})
    short_run_id = summary.run_id[:8]
    sha_preview = summary.dataset_sha256[:16]
    total_runs = len(summary.runs)
    total_samples = len({r.sample.sample_idx for r in aggregated})

    # --- Run metadata header ---
    typer.echo("")
    _rule(color=_CYAN)
    typer.echo(f"{_CYAN}{_BOLD}  k-eval  ·  Run Complete{_RESET}")
    _rule(color=_CYAN)
    typer.echo("")

    meta_rows: list[tuple[str, str]] = [
        ("Run ID", f"{short_run_id}-..."),
        ("Config", summary.config_name),
        ("Dataset SHA256", f"{sha_preview}..."),
        ("Samples", str(total_samples)),
        ("Conditions", ", ".join(conditions)),
        ("Total runs", str(total_runs)),
        ("Elapsed", _format_elapsed(elapsed_seconds=elapsed_seconds)),
        ("Aggregate JSON", str(json_path)),
        ("Detailed JSONL", str(jsonl_path)),
    ]
    label_w = max(len(label) for label, _ in meta_rows)
    for label, value in meta_rows:
        typer.echo(f"  {_DIM}{label:<{label_w}}{_RESET}  {_WHITE}{value}{_RESET}")

    # Metric definitions: (display label, attribute prefix on AggregatedResult).
    metrics: list[tuple[str, str]] = [
        ("Factual Adherence", "factual_adherence"),
        ("Completeness", "completeness"),
        ("Helpfulness & Clarity", "helpfulness_and_clarity"),
    ]
    metric_w = max(len(label) for label, _ in metrics)

    if len(conditions) == 1:
        # Single-condition: the helper owns its own header and rule.
        cond = conditions[0]
        results = [r for r in aggregated if r.condition == cond]
        _print_single_condition(
            condition=cond,
            results=results,
            metrics=metrics,
            metric_w=metric_w,
        )
    else:
        typer.echo("")
        _rule(color=_BLUE)
        typer.echo(f"{_BLUE}{_BOLD}  Results by Condition{_RESET}")
        typer.echo("")
        results_by_condition = {
            c: [r for r in aggregated if r.condition == c] for c in conditions
        }
        _print_comparison_table(
            conditions=conditions,
            results_by_condition=results_by_condition,
            metrics=metrics,
            metric_w=metric_w,
        )

    typer.echo("")
    _rule(color=_CYAN)
    typer.echo("")


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to evaluation config YAML"),
    output_dir: Path = typer.Option(
        Path("./results"),
        "--output-dir",
        "-o",
        help="Directory for output files",
    ),
    log_format: str = typer.Option(
        "console",
        "--log-format",
        help="Log format: 'console' or 'json'",
    ),
) -> None:
    """Run a k-eval evaluation from a YAML config file."""
    try:
        _configure_structlog(log_format=log_format)

        config_observer = StructlogConfigObserver()
        loader = YamlConfigLoader(observer=config_observer)
        try:
            config = loader.load(path=config_path)
        except KEvalError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc

        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_loader = JsonlDatasetLoader(observer=StructlogDatasetObserver())
        agent_factory = create_agent_factory(
            config=config.agent,
            observer=StructlogAgentObserver(),
        )
        judge_factory = LiteLLMJudgeFactory(
            config=config.judge,
            observer=StructlogJudgeObserver(),
        )
        observers: list[EvaluationObserver] = [StructlogEvaluationObserver()]
        if log_format != "json":
            observers.append(ProgressEvaluationObserver())
        evaluation_observer = CompositeEvaluationObserver(observers=observers)

        evaluation_runner = EvaluationRunner(
            config=config,
            dataset_loader=dataset_loader,
            agent_factory=agent_factory,
            judge_factory=judge_factory,
            observer=evaluation_observer,
        )

        started_at = time.monotonic()
        summary: RunSummary = asyncio.run(evaluation_runner.run())
        elapsed_seconds = time.monotonic() - started_at

        aggregated = aggregate(runs=summary.runs)
        stem = _output_stem(config_name=summary.config_name, run_id=summary.run_id)
        json_path, jsonl_path = _write_outputs(
            output_dir=output_dir,
            stem=stem,
            summary=summary,
            aggregated=aggregated,
            agent_config=config.agent,
            judge_config=config.judge,
        )

        _print_summary(
            summary=summary,
            aggregated=aggregated,
            json_path=json_path,
            jsonl_path=jsonl_path,
            elapsed_seconds=elapsed_seconds,
        )

    except KeyboardInterrupt:
        typer.echo("Evaluation interrupted.")
        sys.exit(1)
    except KEvalError as exc:
        typer.echo(str(exc))
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Unexpected error: {exc}\nPlease report this bug.")
        sys.exit(1)


if __name__ == "__main__":
    app()
