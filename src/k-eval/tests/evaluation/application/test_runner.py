"""Tests for EvaluationRunner application logic."""

from pathlib import Path

from agent.domain.result import AgentResult
from agent.domain.usage import UsageMetrics
from config.domain.agent import AgentConfig
from config.domain.condition import ConditionConfig
from config.domain.config import EvalConfig
from config.domain.dataset import DatasetConfig
from config.domain.execution import ExecutionConfig, RetryConfig
from config.domain.judge import JudgeConfig
from dataset.domain.sample import Sample
from evaluation.application.runner import EvaluationRunner
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


def _make_execution_config(num_samples: int = 2) -> ExecutionConfig:
    return ExecutionConfig(
        num_samples=num_samples,
        max_concurrent=1,
        retry=RetryConfig(
            max_attempts=1,
            initial_backoff_seconds=1,
            backoff_multiplier=1,
        ),
    )


def _make_eval_config(
    conditions: dict[str, ConditionConfig],
    num_samples: int = 2,
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
        execution=_make_execution_config(num_samples=num_samples),
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
            num_samples=2,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        results = await runner.run()

        assert len(results) == 8  # 2 samples × 2 conditions × 2 repetitions

    async def test_single_sample_single_condition_single_repetition(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_samples=1,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        results = await runner.run()

        assert len(results) == 1


class TestEvaluationRunnerRunId:
    """run_id is shared across all results from a single run() call."""

    async def test_run_id_is_same_across_all_results(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_samples=2,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(2)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        results = await runner.run()

        run_ids = {r.run_id for r in results}
        assert len(run_ids) == 1

    async def test_run_id_is_a_non_empty_string(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]), num_samples=1
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        results = await runner.run()

        assert results[0].run_id != ""


class TestEvaluationRunnerObserverEvents:
    """Observer receives the correct events in the correct order."""

    async def test_evaluation_started_is_emitted_once(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_samples=3,
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
            num_samples=3,
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
        assert event.num_samples == 3

    async def test_evaluation_completed_is_emitted_once(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_samples=2,
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
            num_samples=2,
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
            num_samples=2,
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


class TestEvaluationRunnerRunIndex:
    """run_index increments correctly within each (sample, condition) group."""

    async def test_run_index_sequence_within_group(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_samples=3,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        results = await runner.run()

        indices = [r.run_index for r in results]
        assert indices == [0, 1, 2]

    async def test_run_index_resets_for_each_condition(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_samples=2,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        results = await runner.run()

        # Results: baseline/0, baseline/1, with-graph/0, with-graph/1
        by_condition: dict[str, list[int]] = {}
        for r in results:
            by_condition.setdefault(r.condition, []).append(r.run_index)

        assert by_condition["baseline"] == [0, 1]
        assert by_condition["with-graph"] == [0, 1]


class TestEvaluationRunnerSampleBinding:
    """EvaluationRun.sample is the correct Sample object."""

    async def test_sample_is_bound_correctly_to_each_run(self) -> None:
        samples = _make_samples(2)
        config = _make_eval_config(
            conditions=_make_conditions(["baseline"]),
            num_samples=1,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=samples),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        results = await runner.run()

        assert results[0].sample is samples[0]
        assert results[1].sample is samples[1]

    async def test_all_results_carry_correct_condition(self) -> None:
        config = _make_eval_config(
            conditions=_make_conditions(["baseline", "with-graph"]),
            num_samples=1,
        )
        runner = EvaluationRunner(
            config=config,
            dataset_loader=FakeDatasetLoader(samples=_make_samples(1)),
            agent_factory=FakeAgentFactory(result=_make_agent_result()),
            judge_factory=FakeJudgeFactory(),
            observer=FakeEvaluationObserver(),
        )

        results = await runner.run()

        conditions = {r.condition for r in results}
        assert conditions == {"baseline", "with-graph"}
