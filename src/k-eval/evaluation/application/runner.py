"""EvaluationRunner — orchestrates the full evaluation loop."""

import uuid

from agent.domain.factory import AgentFactory
from config.domain.config import EvalConfig
from dataset.domain.loader import DatasetLoader
from evaluation.domain.observer import EvaluationObserver
from evaluation.domain.run import EvaluationRun
from judge.domain.factory import JudgeFactory


class EvaluationRunner:
    """Runs the full evaluation: loads samples, loops over conditions, scores results.

    The runner is deliberately free of infrastructure dependencies — it receives
    abstract factories and a loader so that implementations can be swapped for
    testing without touching the orchestration logic.
    """

    def __init__(
        self,
        config: EvalConfig,
        dataset_loader: DatasetLoader,
        agent_factory: AgentFactory,
        judge_factory: JudgeFactory,
        observer: EvaluationObserver,
    ) -> None:
        self._config = config
        self._dataset_loader = dataset_loader
        self._agent_factory = agent_factory
        self._judge_factory = judge_factory
        self._observer = observer

    async def run(self) -> list[EvaluationRun]:
        """Execute the full evaluation and return all EvaluationRun results.

        Iterates: for each sample → for each condition → for each run_index in
        range(num_samples). Agent and judge errors propagate immediately — there
        is no catch-and-continue.
        """
        run_id = str(uuid.uuid4())
        samples = self._dataset_loader.load(config=self._config.dataset)

        self._observer.evaluation_started(
            run_id=run_id,
            total_samples=len(samples),
            total_conditions=len(self._config.conditions),
            num_samples=self._config.execution.num_samples,
        )

        results: list[EvaluationRun] = []

        for sample in samples:
            for condition_name, condition in self._config.conditions.items():
                for run_index in range(self._config.execution.num_samples):
                    self._observer.sample_condition_started(
                        run_id=run_id,
                        sample_idx=sample.sample_idx,
                        condition=condition_name,
                        run_index=run_index,
                    )

                    agent = self._agent_factory.create(
                        condition=condition_name,
                        sample_idx=sample.sample_idx,
                        system_prompt=condition.system_prompt,
                        mcp_servers=condition.mcp_servers,
                    )
                    agent_result = await agent.ask(question=sample.question)

                    judge = self._judge_factory.create(
                        condition=condition_name,
                        sample_idx=sample.sample_idx,
                    )
                    judge_result = await judge.score(
                        question=sample.question,
                        golden_answer=sample.answer,
                        agent_response=agent_result.response,
                    )

                    results.append(
                        EvaluationRun(
                            run_id=run_id,
                            sample=sample,
                            condition=condition_name,
                            run_index=run_index,
                            agent_result=agent_result,
                            judge_result=judge_result,
                        )
                    )

                    self._observer.sample_condition_completed(
                        run_id=run_id,
                        sample_idx=sample.sample_idx,
                        condition=condition_name,
                        run_index=run_index,
                    )

        self._observer.evaluation_completed(
            run_id=run_id,
            total_runs=len(results),
        )

        return results
