"""CLI entrypoint for k-eval — typer app with a `run` command."""

import asyncio
import json
import sys
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
from evaluation.domain.summary import RunSummary
from evaluation.infrastructure.observer import StructlogEvaluationObserver
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


def _score_color(score: float) -> str:
    if score >= 4.0:
        return _GREEN
    if score >= 3.0:
        return _YELLOW
    return _RED


def _rule(width: int = 72, color: str = _DIM) -> None:
    typer.echo(f"{color}{'─' * width}{_RESET}")


def _print_summary(
    summary: RunSummary,
    aggregated: list[AggregatedResult],
    json_path: Path,
    jsonl_path: Path,
) -> None:
    """Print a colorized table summary to stdout."""
    conditions = sorted({r.condition for r in aggregated})
    short_run_id = summary.run_id[:8]
    sha_preview = summary.dataset_sha256[:16]
    total_runs = len(summary.runs)
    total_samples = len({r.sample.sample_idx for r in aggregated})

    # --- Run metadata table ---
    typer.echo("")
    _rule(color=_CYAN)
    typer.echo(f"{_CYAN}{_BOLD}  k-eval  ·  Run Complete{_RESET}")
    _rule(color=_CYAN)
    typer.echo("")

    meta_rows = [
        ("Run ID", f"{short_run_id}-..."),
        ("Config", summary.config_name),
        ("Dataset SHA256", f"{sha_preview}..."),
        ("Samples", str(total_samples)),
        ("Conditions", ", ".join(conditions)),
        ("Total runs", str(total_runs)),
        ("Aggregate JSON", str(json_path)),
        ("Detailed JSONL", str(jsonl_path)),
    ]
    label_w = max(len(label) for label, _ in meta_rows)
    for label, value in meta_rows:
        typer.echo(f"  {_DIM}{label:<{label_w}}{_RESET}  {_WHITE}{value}{_RESET}")

    # --- Per-condition scores table ---
    metrics = [
        ("Factual Adherence", "factual_adherence"),
        ("Completeness", "completeness"),
        ("Helpfulness & Clarity", "helpfulness_and_clarity"),
    ]
    metric_w = max(len(label) for label, _ in metrics)

    for condition in conditions:
        condition_results = [r for r in aggregated if r.condition == condition]
        n = len(condition_results)

        typer.echo("")
        _rule(color=_BLUE)
        typer.echo(f"{_BLUE}{_BOLD}  {condition}{_RESET}")
        _rule(color=_BLUE)
        typer.echo(
            f"  {_DIM}{'Metric':<{metric_w}}  {'Mean':>6}  {'StdDev':>6}  {'Bar'}{_RESET}"
        )
        typer.echo(f"  {'─' * metric_w}  {'─' * 6}  {'─' * 6}  {'─' * 10}")

        for label, attr in metrics:
            mean = sum(getattr(r, f"{attr}_mean") for r in condition_results) / n
            std = sum(getattr(r, f"{attr}_stddev") for r in condition_results) / n
            color = _score_color(mean)
            filled = round(mean)
            bar = f"{color}{'█' * filled}{_DIM}{'░' * (5 - filled)}{_RESET}"
            typer.echo(
                f"  {_WHITE}{label:<{metric_w}}{_RESET}"
                f"  {color}{mean:>6.2f}{_RESET}"
                f"  {_DIM}{std:>6.2f}{_RESET}"
                f"  {bar}"
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
        evaluation_observer = StructlogEvaluationObserver()

        evaluation_runner = EvaluationRunner(
            config=config,
            dataset_loader=dataset_loader,
            agent_factory=agent_factory,
            judge_factory=judge_factory,
            observer=evaluation_observer,
        )

        summary: RunSummary = asyncio.run(evaluation_runner.run())

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
