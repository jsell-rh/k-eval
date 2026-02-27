"""Tests for EvaluationRunner application logic."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from k_eval.agent.domain.result import AgentResult
from k_eval.agent.domain.turn import AgentTurn, ToolCall
from k_eval.agent.domain.usage import UsageMetrics
from k_eval.agent.infrastructure.errors import (
    AgentInvocationError,
    McpToolSuccessAbsentError,
    McpToolUseAbsentError,
)
from k_eval.config.domain.agent import AgentConfig
from k_eval.config.domain.condition import ConditionConfig
from k_eval.config.domain.condition_mcp_server import ConditionMcpServer
from k_eval.config.domain.config import EvalConfig
from k_eval.config.domain.dataset import DatasetConfig
from k_eval.config.domain.execution import ExecutionConfig, RetryConfig
from k_eval.config.domain.judge import JudgeConfig
from k_eval.core.errors import KEvalError
from k_eval.dataset.domain.sample import Sample
from k_eval.evaluation.application.runner import EvaluationRunner
from k_eval.evaluation.domain.summary import RunSummary
from tests.agent.fake_agent import FakeAgent
from tests.agent.fake_factory import FakeAgentFactory
from tests.evaluation.fake_dataset_loader import FakeDatasetLoader
from tests.evaluation.fake_observer import FakeEvaluationObserver
from tests.judge.fake_factory import FakeJudgeFactory


def _make_agent_result() -> AgentResult:
    return AgentResult(
        response="The answer is 42.",
        cost_usd=0.001,
        duration_ms=500,
        duration_api_ms=400,
        num_turns=1,
        usage=UsageMetrics(input_tokens=50, output_tokens=20),
    )


def _make_execution_config(num_repetitions: int = 2) -> ExecutionConfig:
    return ExecutionConfig(
        num_repetitions=num_repetitions,
        max_concurrent=1,
        retry=RetryConfig(
            max_attempts=1,
            initial_backoff_seconds=1,
            backoff_multiplier=1,
        ),
    )


def _make_eval_config(
    conditions: dict[str, ConditionConfig],
    num_repetitions: int = 2,
) -> EvalConfig:
    return EvalConfig(
        name="test-run",
        version="1.0",
        dataset=DatasetConfig(
            path=Path("/dev/null"),
            question_key="question",
            answer_key="answer",
        ),
        agent=AgentConfig(type="claude", model="claude-3-5-sonnet"),
        judge=JudgeConfig(model="gpt-4o", temperature=0.0),
        mcp_servers={},
        conditions=conditions,
        execution=_make_execution_config(num_repetitions=num_repetitions),
    )


def _make_conditions(names: list[str]) -> dict[str, ConditionConfig]:
    return {
        name: ConditionConfig(mcp_servers=[], system_prompt=f"You are {name}.")
        for name in names
    }


def _make_samples(count: int) -> list[Sample]:
    return [
        Sample(sample_idx=f"s{i}", question=f"Question {i}?", answer=f"Answer {i}.")
        for i in range(count)
    ]


class TestEvaluationRunnerResultCount:
    """Runner produces the correct number of EvaluationRun objects."""

    async def test_two_samples_two_conditions_two_repetitions_produces_eight_runs(
        self,
    ) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_repetitions=2,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert len(result.runs) == 8  # 2 samples × 2 conditions × 2 repetitions

    async def test_single_sample_single_condition_single_repetition(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=1,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert len(result.runs) == 1


class TestEvaluationRunnerRunSummary:
    """RunSummary fields are populated correctly."""

    async def test_run_id_is_non_empty(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=1,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert result.run_id != ""

    async def test_dataset_sha256_matches_fake_loader(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=1,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(
                samples=_make_samples(1), sha256="fake-sha256"
            ),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert result.dataset_sha256 == "fake-sha256"

    async def test_config_name_matches_eval_config(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=1,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert result.config_name == config.name

    async def test_returns_run_summary_instance(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=1,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert isinstance(result, RunSummary)


class TestEvaluationRunnerRunId:
    """run_id is shared across all results from a single run() call."""

    async def test_run_id_is_same_across_all_results(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_repetitions=2,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        run_ids = {r.run_id for r in result.runs}
        assert len(run_ids) == 1

    async def test_run_id_is_a_non_empty_string(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]), num_repetitions=1
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert result.runs[0].run_id != ""


class TestEvaluationRunnerObserverEvents:
    """Observer receives the correct events in the correct order."""

    async def test_evaluation_started_is_emitted_once(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_repetitions=3,
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        assert len(observer.started) == 1

    async def test_evaluation_started_carries_correct_counts(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_repetitions=3,
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        event = observer.started[0]
        assert event.total_samples == 2
        assert event.total_conditions == 2
        assert event.num_repetitions == 3
        assert event.max_concurrent == config.execution.max_concurrent

    async def test_evaluation_completed_is_emitted_once(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=2,
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        assert len(observer.completed) == 1

    async def test_evaluation_completed_carries_total_runs(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_repetitions=2,
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        assert observer.completed[0].total_runs == 8  # 2 × 2 × 2

    async def test_sample_condition_started_and_completed_emitted_for_each_triple(
        self,
    ) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_repetitions=2,
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        # 2 samples × 2 conditions × 2 repetitions = 8 each
        assert len(observer.sc_started) == 8
        assert len(observer.sc_completed) == 8


class TestEvaluationRunnerRepetitionIndex:
    """repetition_index increments correctly within each (sample, condition) group."""

    async def test_repetition_index_sequence_within_group(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=3,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        indices = [r.repetition_index for r in result.runs]
        assert indices == [0, 1, 2]

    async def test_repetition_index_resets_for_each_condition(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_repetitions=2,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        # Results: baseline/0, baseline/1, with-graph/0, with-graph/1
        by_condition: dict[str, list[int]] = {}
        for r in result.runs:
            by_condition.setdefault(r.condition, []).append(r.repetition_index)

        assert by_condition["baseline"] == [0, 1]
        assert by_condition["with-graph"] == [0, 1]


class TestEvaluationRunnerSampleBinding:
    """EvaluationRun.sample is the correct Sample object."""

    async def test_sample_is_bound_correctly_to_each_run(self) -> None:
        samples = _make_samples(2)
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=1,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=samples),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        run_by_sample = {r.sample.sample_idx: r for r in result.runs}
        assert run_by_sample[samples[0].sample_idx].sample is samples[0]
        assert run_by_sample[samples[1].sample_idx].sample is samples[1]

    async def test_all_results_carry_correct_condition(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_repetitions=1,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        conditions = {r.condition for r in result.runs}
        assert conditions == {"baseline", "with-graph"}


def _make_retry_config(
    max_attempts: int,
    initial_backoff_seconds: int = 1,
    backoff_multiplier: int = 2,
) -> ExecutionConfig:
    return ExecutionConfig(
        num_repetitions=1,
        max_concurrent=1,
        retry=RetryConfig(
            max_attempts=max_attempts,
            initial_backoff_seconds=initial_backoff_seconds,
            backoff_multiplier=backoff_multiplier,
        ),
    )


def _make_retry_eval_config(execution: ExecutionConfig) -> EvalConfig:
    return EvalConfig(
        name="retry-test",
        version="1.0",
        dataset=DatasetConfig(
            path=Path("/dev/null"),
            question_key="question",
            answer_key="answer",
        ),
        agent=AgentConfig(type="claude", model="claude-3-5-sonnet"),
        judge=JudgeConfig(model="gpt-4o", temperature=0.0),
        mcp_servers={},
        conditions=_make_conditions(["baseline"]),
        execution=execution,
    )


class TestEvaluationRunnerRetry:
    """Runner retry behaviour: backoff, retry events, and abort after max_attempts."""

    @patch("k_eval.evaluation.application.runner.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_retriable_agent_error(
        self, mock_sleep: AsyncMock
    ) -> None:
        """First agent call raises a retriable error; second succeeds. Runner completes."""
        config = _make_retry_eval_config(
            execution=_make_retry_config(max_attempts=3),
        )
        observer = FakeEvaluationObserver()
        agent_result = _make_agent_result()
        failing_agent = FakeAgent(
            result=agent_result,
            side_effects=[AgentInvocationError(reason="rate limit", retriable=True)],
        )
        succeeding_agent = FakeAgent(result=agent_result)

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(
                result=agent_result,
                agents=[failing_agent, succeeding_agent],
            ),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        result = await runner.run()

        assert len(result.runs) == 1
        assert len(observer.sc_retried) == 1
        assert observer.sc_retried[0].attempt == 1

    @patch("k_eval.evaluation.application.runner.asyncio.sleep", new_callable=AsyncMock)
    async def test_does_not_retry_on_non_retriable_error(
        self, mock_sleep: AsyncMock
    ) -> None:
        """A non-retriable error propagates immediately without any retry."""
        config = _make_retry_eval_config(
            execution=_make_retry_config(max_attempts=3),
        )
        observer = FakeEvaluationObserver()
        agent_result = _make_agent_result()
        failing_agent = FakeAgent(
            result=agent_result,
            side_effects=[AgentInvocationError(reason="bad config", retriable=False)],
        )

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(
                result=agent_result,
                agents=[failing_agent],
            ),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        with pytest.raises(AgentInvocationError):
            await runner.run()

        assert len(observer.sc_retried) == 0
        mock_sleep.assert_not_called()

    @patch("k_eval.evaluation.application.runner.asyncio.sleep", new_callable=AsyncMock)
    async def test_aborts_after_max_attempts(self, mock_sleep: AsyncMock) -> None:
        """Runner raises after exhausting all max_attempts. Observer sees max_attempts-1 retries."""
        max_attempts = 3
        config = _make_retry_eval_config(
            execution=_make_retry_config(max_attempts=max_attempts),
        )
        observer = FakeEvaluationObserver()
        agent_result = _make_agent_result()

        # All agents always raise a retriable error.
        always_failing_agents = [
            FakeAgent(
                result=agent_result,
                side_effects=[
                    AgentInvocationError(reason="rate limit", retriable=True)
                ],
            )
            for _ in range(max_attempts)
        ]

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(
                result=agent_result,
                agents=always_failing_agents,
            ),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        with pytest.raises(AgentInvocationError):
            await runner.run()

        # Attempts 1 and 2 emit retry events; attempt 3 raises.
        assert len(observer.sc_retried) == max_attempts - 1
        assert observer.sc_retried[0].attempt == 1
        assert observer.sc_retried[1].attempt == 2

    @patch("k_eval.evaluation.application.runner.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_emits_correct_backoff(self, mock_sleep: AsyncMock) -> None:
        """Backoff values in retry events follow initial_backoff * multiplier^n."""
        config = _make_retry_eval_config(
            execution=_make_retry_config(
                max_attempts=4,
                initial_backoff_seconds=2,
                backoff_multiplier=3,
            ),
        )
        observer = FakeEvaluationObserver()
        agent_result = _make_agent_result()

        # 3 failing agents followed by a successful one.
        agents: list[FakeAgent] = [
            FakeAgent(
                result=agent_result,
                side_effects=[
                    AgentInvocationError(reason="rate limit", retriable=True)
                ],
            )
            for _ in range(3)
        ]
        agents.append(FakeAgent(result=agent_result))

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=agent_result, agents=agents),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        assert len(observer.sc_retried) == 3
        # attempt 1: backoff = 2 * 3^0 = 2
        assert observer.sc_retried[0].backoff_seconds == 2.0
        # attempt 2: backoff = 2 * 3^1 = 6
        assert observer.sc_retried[1].backoff_seconds == 6.0
        # attempt 3: backoff = 2 * 3^2 = 18
        assert observer.sc_retried[2].backoff_seconds == 18.0


# ---------------------------------------------------------------------------
# Concurrency tracking helpers
# ---------------------------------------------------------------------------


def _make_default_agent_result() -> AgentResult:
    return AgentResult(
        response="The answer is 42.",
        cost_usd=0.001,
        duration_ms=500,
        duration_api_ms=400,
        num_turns=1,
        usage=UsageMetrics(input_tokens=50, output_tokens=20),
    )


class ConcurrencyTracker:
    """Records peak concurrent usage across async tasks."""

    def __init__(self) -> None:
        self.peak = 0
        self._active = 0

    async def enter(self) -> None:
        self._active += 1
        self.peak = max(self.peak, self._active)

    async def exit(self) -> None:
        self._active -= 1


class ConcurrencyTrackingFakeAgent:
    """Satisfies the Agent protocol. Tracks concurrent ask() calls via a ConcurrencyTracker."""

    def __init__(self, tracker: ConcurrencyTracker) -> None:
        self._tracker = tracker

    async def ask(self, question: str) -> AgentResult:
        await self._tracker.enter()
        await asyncio.sleep(0.01)  # yield to allow other tasks to start
        await self._tracker.exit()
        return _make_default_agent_result()


class ConcurrencyTrackingFakeAgentFactory:
    """Satisfies the AgentFactory protocol. Returns ConcurrencyTrackingFakeAgent instances."""

    def __init__(self, tracker: ConcurrencyTracker) -> None:
        self._tracker = tracker

    def create(
        self,
        condition: str,
        sample_idx: str,
        system_prompt: str,
        mcp_servers: list[ConditionMcpServer],
    ) -> ConcurrencyTrackingFakeAgent:
        return ConcurrencyTrackingFakeAgent(tracker=self._tracker)


def _make_concurrent_eval_config(
    num_dataset_samples: int,
    num_conditions: int,
    num_repetitions: int,
    max_concurrent: int,
) -> EvalConfig:
    conditions = _make_conditions([f"cond-{i}" for i in range(num_conditions)])
    return EvalConfig(
        name="concurrency-test",
        version="1.0",
        dataset=DatasetConfig(
            path=Path("/dev/null"),
            question_key="question",
            answer_key="answer",
        ),
        agent=AgentConfig(type="claude", model="claude-3-5-sonnet"),
        judge=JudgeConfig(model="gpt-4o", temperature=0.0),
        mcp_servers={},
        conditions=conditions,
        execution=ExecutionConfig(
            num_repetitions=num_repetitions,
            max_concurrent=max_concurrent,
            retry=RetryConfig(
                max_attempts=1,
                initial_backoff_seconds=0,
                backoff_multiplier=1,
            ),
        ),
    )


class TestEvaluationRunnerConcurrency:
    """Concurrency behaviour: semaphore limits parallelism and all triples complete."""

    async def test_max_concurrent_limits_peak_concurrency(self) -> None:
        """10 samples × 1 condition × 1 repetition with max_concurrent=3 must never exceed 3 simultaneous agent calls."""
        tracker = ConcurrencyTracker()
        config = _make_concurrent_eval_config(
            num_dataset_samples=10,
            num_conditions=1,
            num_repetitions=1,
            max_concurrent=3,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(10)),
            agent_factory=ConcurrencyTrackingFakeAgentFactory(tracker=tracker),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        await runner.run()

        assert tracker.peak <= 3

    async def test_all_triples_present_in_results(self) -> None:
        """3 samples × 2 conditions × 2 repetitions with max_concurrent=4 produces all 12 triples."""
        config = _make_concurrent_eval_config(
            num_dataset_samples=3,
            num_conditions=2,
            num_repetitions=2,
            max_concurrent=4,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(3)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert len(result.runs) == 12  # 3 × 2 × 2
        triples = {
            (r.sample.sample_idx, r.condition, r.repetition_index) for r in result.runs
        }
        assert len(triples) == 12

    async def test_results_sorted_deterministically(self) -> None:
        """Results are sorted by (sample_idx, condition, repetition_index) after concurrent execution."""
        config = _make_concurrent_eval_config(
            num_dataset_samples=3,
            num_conditions=2,
            num_repetitions=2,
            max_concurrent=4,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(3)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        sort_keys = [
            (r.sample.sample_idx, r.condition, r.repetition_index) for r in result.runs
        ]
        assert sort_keys == sorted(sort_keys)

    async def test_non_retriable_error_raises_keval_error_not_exception_group(
        self,
    ) -> None:
        """A non-retriable AgentInvocationError surfaces as KEvalError, not BaseExceptionGroup."""
        config = _make_concurrent_eval_config(
            num_dataset_samples=1,
            num_conditions=1,
            num_repetitions=1,
            max_concurrent=1,
        )
        failing_agent = FakeAgent(
            result=_make_agent_result(),
            side_effects=[AgentInvocationError(reason="bad config", retriable=False)],
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(
                result=_make_agent_result(),
                agents=[failing_agent],
            ),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        with pytest.raises(KEvalError) as exc_info:
            await runner.run()

        assert not isinstance(exc_info.value, BaseExceptionGroup)


# ---------------------------------------------------------------------------
# Progress and elapsed time
# ---------------------------------------------------------------------------


class TestEvaluationRunnerProgress:
    """evaluation_progress is emitted once per resolved triple with correct counters."""

    async def test_progress_emitted_once_per_triple(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_repetitions=2,
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        # 2 samples × 2 conditions × 2 repetitions = 8 triples → 8 progress events
        assert len(observer.progress) == 8
        for event in observer.progress:
            assert event.condition in {"baseline", "with-graph"}

    async def test_progress_total_matches_triple_count(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=3,
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        # All events should report the same total: 2 × 1 × 3 = 6
        assert all(e.total == 6 for e in observer.progress)

    async def test_progress_completed_values_are_unique_and_sequential(self) -> None:
        """Each progress event increments completed by exactly 1."""
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=3,
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        completed_values = sorted(e.completed for e in observer.progress)
        assert completed_values == [1, 2, 3]

    async def test_progress_emitted_on_permanent_failure(self) -> None:
        """A non-retriable failure still emits a progress event (triple resolved)."""
        config = _make_concurrent_eval_config(
            num_dataset_samples=1,
            num_conditions=1,
            num_repetitions=1,
            max_concurrent=1,
        )
        failing_agent = FakeAgent(
            result=_make_agent_result(),
            side_effects=[AgentInvocationError(reason="bad config", retriable=False)],
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(
                result=_make_agent_result(),
                agents=[failing_agent],
            ),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        with pytest.raises(KEvalError):
            await runner.run()

        assert len(observer.progress) == 1
        assert observer.progress[0].completed == 1
        assert observer.progress[0].total == 1
        assert observer.progress[0].condition in {"cond-0"}

    async def test_run_id_consistent_across_progress_events(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=2,
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        run_ids = {e.run_id for e in observer.progress}
        assert len(run_ids) == 1
        assert run_ids.pop() == observer.started[0].run_id


class TestEvaluationRunnerElapsed:
    """evaluation_completed carries a positive elapsed_seconds."""

    async def test_elapsed_seconds_is_positive(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=1,
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        assert observer.completed[0].elapsed_seconds >= 0.0

    async def test_elapsed_seconds_is_a_float(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_repetitions=1,
        )
        observer = FakeEvaluationObserver()
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        await runner.run()

        assert isinstance(observer.completed[0].elapsed_seconds, float)


# ---------------------------------------------------------------------------
# MCP tool use enforcement gate
# ---------------------------------------------------------------------------


def _make_tool_call(tool_use_id: str = "tu-1", tool_name: str = "search") -> ToolCall:
    return ToolCall(
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        tool_input={"q": "test"},
        tool_result="result",
        tool_error=False,
    )


def _make_agent_result_with_tool_calls() -> AgentResult:
    tc = _make_tool_call()
    tool_turn = AgentTurn(
        turn_idx=0,
        role="tool_use",
        text=None,
        tool_calls=[tc],
    )
    return AgentResult(
        response="The answer.",
        cost_usd=0.001,
        duration_ms=500,
        duration_api_ms=400,
        num_turns=2,
        usage=UsageMetrics(input_tokens=50, output_tokens=20),
        turns=[tool_turn],
    )


def _make_agent_result_without_tool_calls() -> AgentResult:
    """AgentResult with only an assistant text turn and no tool calls."""
    text_turn = AgentTurn(
        turn_idx=0,
        role="assistant",
        text="I don't need to use any tools.",
        tool_calls=[],
    )
    return AgentResult(
        response="The answer without tools.",
        cost_usd=0.001,
        duration_ms=500,
        duration_api_ms=400,
        num_turns=1,
        usage=UsageMetrics(input_tokens=50, output_tokens=20),
        turns=[text_turn],
    )


def _make_condition_requiring_tool_use() -> ConditionConfig:
    return ConditionConfig(
        mcp_servers=[],
        system_prompt="You must use MCP tools.",
        require_mcp_tool_use=True,
    )


def _make_condition_not_requiring_tool_use() -> ConditionConfig:
    return ConditionConfig(
        mcp_servers=[],
        system_prompt="You are a helpful assistant.",
        require_mcp_tool_use=False,
    )


def _make_mcp_tool_required_eval_config() -> EvalConfig:
    return EvalConfig(
        name="mcp-test",
        version="1.0",
        dataset=DatasetConfig(
            path=Path("/dev/null"),
            question_key="question",
            answer_key="answer",
        ),
        agent=AgentConfig(type="claude", model="claude-3-5-sonnet"),
        judge=JudgeConfig(model="gpt-4o", temperature=0.0),
        mcp_servers={},
        conditions={"with-tools": _make_condition_requiring_tool_use()},
        execution=ExecutionConfig(
            num_repetitions=1,
            max_concurrent=1,
            retry=RetryConfig(
                max_attempts=1,
                initial_backoff_seconds=1,
                backoff_multiplier=1,
            ),
        ),
    )


class TestMcpToolUseEnforcementGate:
    """When require_mcp_tool_use is True, runner raises McpToolUseAbsentError if no tools called."""

    async def test_raises_when_no_tool_calls_and_required(self) -> None:
        config = _make_mcp_tool_required_eval_config()
        agent_result = _make_agent_result_without_tool_calls()

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=agent_result),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        with pytest.raises(McpToolUseAbsentError):
            await runner.run()

    async def test_does_not_raise_when_tool_calls_present(self) -> None:
        config = _make_mcp_tool_required_eval_config()
        agent_result = _make_agent_result_with_tool_calls()

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=agent_result),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert len(result.runs) == 1

    async def test_does_not_raise_when_tool_use_not_required(self) -> None:
        """require_mcp_tool_use=False means no enforcement even with no tool calls."""
        config = EvalConfig(
            name="no-requirement",
            version="1.0",
            dataset=DatasetConfig(
                path=Path("/dev/null"),
                question_key="question",
                answer_key="answer",
            ),
            agent=AgentConfig(type="claude", model="claude-3-5-sonnet"),
            judge=JudgeConfig(model="gpt-4o", temperature=0.0),
            mcp_servers={},
            conditions={"baseline": _make_condition_not_requiring_tool_use()},
            execution=ExecutionConfig(
                num_repetitions=1,
                max_concurrent=1,
                retry=RetryConfig(
                    max_attempts=1,
                    initial_backoff_seconds=1,
                    backoff_multiplier=1,
                ),
            ),
        )
        agent_result = _make_agent_result_without_tool_calls()

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=agent_result),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert len(result.runs) == 1

    async def test_mcp_tool_use_absent_is_retriable(self) -> None:
        """McpToolUseAbsentError is retriable — should trigger retry behavior."""
        config = EvalConfig(
            name="retry-mcp",
            version="1.0",
            dataset=DatasetConfig(
                path=Path("/dev/null"),
                question_key="question",
                answer_key="answer",
            ),
            agent=AgentConfig(type="claude", model="claude-3-5-sonnet"),
            judge=JudgeConfig(model="gpt-4o", temperature=0.0),
            mcp_servers={},
            conditions={"with-tools": _make_condition_requiring_tool_use()},
            execution=ExecutionConfig(
                num_repetitions=1,
                max_concurrent=1,
                retry=RetryConfig(
                    max_attempts=2,
                    initial_backoff_seconds=0,
                    backoff_multiplier=1,
                ),
            ),
        )
        # First agent call returns no tool calls; second returns tool calls.
        no_tools_agent = FakeAgent(result=_make_agent_result_without_tool_calls())
        with_tools_agent = FakeAgent(result=_make_agent_result_with_tool_calls())
        observer = FakeEvaluationObserver()

        with patch(
            "k_eval.evaluation.application.runner.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            runner = EvaluationRunner(
                config=config,
                dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
                agent_factory=FakeAgentFactory(
                    result=_make_agent_result_with_tool_calls(),
                    agents=[no_tools_agent, with_tools_agent],
                ),
                judge_factory=FakeJudgeFactory(),
                observer=observer,
            )

            result = await runner.run()

        assert len(result.runs) == 1
        assert len(observer.sc_retried) == 1

    async def test_observer_mcp_tool_use_absent_emitted(self) -> None:
        """mcp_tool_use_absent observer event is emitted before raising."""
        config = _make_mcp_tool_required_eval_config()
        agent_result = _make_agent_result_without_tool_calls()
        observer = FakeEvaluationObserver()

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=agent_result),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        with pytest.raises(McpToolUseAbsentError):
            await runner.run()

        assert len(observer.mcp_absent) == 1
        event = observer.mcp_absent[0]
        assert event.condition == "with-tools"


# ---------------------------------------------------------------------------
# MCP tool success enforcement gate
# ---------------------------------------------------------------------------


def _make_agent_result_with_all_errored_tool_calls() -> AgentResult:
    """AgentResult where every tool call has tool_error=True."""
    tc = ToolCall(
        tool_use_id="tu-err",
        tool_name="failing_tool",
        tool_input={},
        tool_result="Error: something broke.",
        tool_error=True,
    )
    tool_turn = AgentTurn(
        turn_idx=0,
        role="tool_use",
        text=None,
        tool_calls=[tc],
    )
    return AgentResult(
        response="I could not get the information.",
        cost_usd=0.001,
        duration_ms=500,
        duration_api_ms=400,
        num_turns=2,
        usage=UsageMetrics(input_tokens=50, output_tokens=20),
        turns=[tool_turn],
    )


def _make_condition_requiring_tool_success() -> ConditionConfig:
    return ConditionConfig(
        mcp_servers=[],
        system_prompt="You must use MCP tools successfully.",
        require_mcp_tool_use=True,
        require_mcp_tool_success=True,
    )


def _make_mcp_tool_success_required_eval_config() -> EvalConfig:
    return EvalConfig(
        name="mcp-success-test",
        version="1.0",
        dataset=DatasetConfig(
            path=Path("/dev/null"),
            question_key="question",
            answer_key="answer",
        ),
        agent=AgentConfig(type="claude", model="claude-3-5-sonnet"),
        judge=JudgeConfig(model="gpt-4o", temperature=0.0),
        mcp_servers={},
        conditions={"with-tools": _make_condition_requiring_tool_success()},
        execution=ExecutionConfig(
            num_repetitions=1,
            max_concurrent=1,
            retry=RetryConfig(
                max_attempts=1,
                initial_backoff_seconds=1,
                backoff_multiplier=1,
            ),
        ),
    )


class TestMcpToolSuccessEnforcementGate:
    """When require_mcp_tool_success is True, runner raises McpToolSuccessAbsentError
    if all tool calls errored."""

    async def test_raises_when_all_tool_calls_errored(self) -> None:
        config = _make_mcp_tool_success_required_eval_config()
        agent_result = _make_agent_result_with_all_errored_tool_calls()

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=agent_result),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        with pytest.raises(McpToolSuccessAbsentError):
            await runner.run()

    async def test_does_not_raise_when_at_least_one_tool_call_succeeded(self) -> None:
        config = _make_mcp_tool_success_required_eval_config()
        agent_result = _make_agent_result_with_tool_calls()  # tool_error=False

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=agent_result),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert len(result.runs) == 1

    async def test_does_not_raise_when_require_mcp_tool_success_is_false(
        self,
    ) -> None:
        """require_mcp_tool_success=False means all-error tool calls are allowed."""
        config = EvalConfig(
            name="no-success-requirement",
            version="1.0",
            dataset=DatasetConfig(
                path=Path("/dev/null"),
                question_key="question",
                answer_key="answer",
            ),
            agent=AgentConfig(type="claude", model="claude-3-5-sonnet"),
            judge=JudgeConfig(model="gpt-4o", temperature=0.0),
            mcp_servers={},
            conditions={
                "with-tools": ConditionConfig(
                    mcp_servers=[],
                    system_prompt="You are a helpful assistant.",
                    require_mcp_tool_use=False,
                    require_mcp_tool_success=False,
                )
            },
            execution=ExecutionConfig(
                num_repetitions=1,
                max_concurrent=1,
                retry=RetryConfig(
                    max_attempts=1,
                    initial_backoff_seconds=1,
                    backoff_multiplier=1,
                ),
            ),
        )
        agent_result = _make_agent_result_with_all_errored_tool_calls()

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=agent_result),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        result = await runner.run()

        assert len(result.runs) == 1

    async def test_gate_does_not_fire_when_no_tool_calls(self) -> None:
        """require_mcp_tool_success gate only fires if there ARE tool calls but all errored.
        If zero tool calls, that is handled by require_mcp_tool_use gate (or not at all)."""
        config = EvalConfig(
            name="success-gate-no-calls",
            version="1.0",
            dataset=DatasetConfig(
                path=Path("/dev/null"),
                question_key="question",
                answer_key="answer",
            ),
            agent=AgentConfig(type="claude", model="claude-3-5-sonnet"),
            judge=JudgeConfig(model="gpt-4o", temperature=0.0),
            mcp_servers={},
            conditions={
                "with-tools": ConditionConfig(
                    mcp_servers=[],
                    system_prompt="You are a helpful assistant.",
                    require_mcp_tool_use=False,
                    require_mcp_tool_success=True,
                )
            },
            execution=ExecutionConfig(
                num_repetitions=1,
                max_concurrent=1,
                retry=RetryConfig(
                    max_attempts=1,
                    initial_backoff_seconds=1,
                    backoff_multiplier=1,
                ),
            ),
        )
        agent_result = _make_agent_result_without_tool_calls()

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=agent_result),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        # No tool calls at all — success gate should NOT fire.
        result = await runner.run()

        assert len(result.runs) == 1

    async def test_observer_mcp_tool_success_absent_emitted(self) -> None:
        """mcp_tool_success_absent observer event is emitted before raising."""
        config = _make_mcp_tool_success_required_eval_config()
        agent_result = _make_agent_result_with_all_errored_tool_calls()
        observer = FakeEvaluationObserver()

        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=agent_result),
            judge_factory=FakeJudgeFactory(),
            observer=observer,
        )

        with pytest.raises(McpToolSuccessAbsentError):
            await runner.run()

        assert len(observer.mcp_success_absent) == 1
        event = observer.mcp_success_absent[0]
        assert event.condition == "with-tools"

    async def test_mcp_tool_success_absent_is_retriable(self) -> None:
        """McpToolSuccessAbsentError is retriable — should trigger retry behavior."""
        config = EvalConfig(
            name="retry-mcp-success",
            version="1.0",
            dataset=DatasetConfig(
                path=Path("/dev/null"),
                question_key="question",
                answer_key="answer",
            ),
            agent=AgentConfig(type="claude", model="claude-3-5-sonnet"),
            judge=JudgeConfig(model="gpt-4o", temperature=0.0),
            mcp_servers={},
            conditions={"with-tools": _make_condition_requiring_tool_success()},
            execution=ExecutionConfig(
                num_repetitions=1,
                max_concurrent=1,
                retry=RetryConfig(
                    max_attempts=2,
                    initial_backoff_seconds=0,
                    backoff_multiplier=1,
                ),
            ),
        )
        # First call: all tools errored; second call: tool succeeds.
        errored_agent = FakeAgent(
            result=_make_agent_result_with_all_errored_tool_calls()
        )
        success_agent = FakeAgent(result=_make_agent_result_with_tool_calls())
        observer = FakeEvaluationObserver()

        with patch(
            "k_eval.evaluation.application.runner.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            runner = EvaluationRunner(
                config=config,
                dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
                agent_factory=FakeAgentFactory(
                    result=_make_agent_result_with_tool_calls(),
                    agents=[errored_agent, success_agent],
                ),
                judge_factory=FakeJudgeFactory(),
                observer=observer,
            )

            result = await runner.run()

        assert len(result.runs) == 1
        assert len(observer.sc_retried) == 1
