"""EvaluationRunner — orchestrates the full evaluation loop."""

import asyncio
import time
import uuid

from agent.domain.factory import AgentFactory
from config.domain.condition import ConditionConfig
from config.domain.config import EvalConfig
from core.errors import KEvalError
from dataset.domain.loader import DatasetLoader
from dataset.domain.sample import Sample
from evaluation.domain.observer import EvaluationObserver
from evaluation.domain.run import EvaluationRun
from evaluation.domain.summary import RunSummary
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

    async def run(self) -> RunSummary:
        """Execute the full evaluation and return a RunSummary.

        Runs all (sample, condition, repetition_index) triples concurrently, bounded by
        max_concurrent. Non-retriable errors (or retry-exhausted errors) abort the
        entire run. On success, results are sorted deterministically by
        (sample_idx, condition, repetition_index) before being returned.
        """
        run_id = str(uuid.uuid4())
        load_result = self._dataset_loader.load(config=self._config.dataset)
        samples = load_result.samples

        self._observer.evaluation_started(
            run_id=run_id,
            total_samples=len(samples),
            total_conditions=len(self._config.conditions),
            condition_names=list(self._config.conditions.keys()),
            num_repetitions=self._config.execution.num_repetitions,
            max_concurrent=self._config.execution.max_concurrent,
        )
        started_at = time.monotonic()

        results: list[EvaluationRun] = []
        sem = asyncio.Semaphore(self._config.execution.max_concurrent)
        total_triples = (
            len(samples)
            * len(self._config.conditions)
            * self._config.execution.num_repetitions
        )
        # Mutable counter shared across concurrent tasks.  The asyncio.Lock
        # makes the read-increment-emit sequence atomic, which is correct even
        # though CPython's GIL would protect a bare int increment — using an
        # explicit lock makes the intent clear and is safe if the runner is
        # ever adapted to use threads.
        completed_count: list[int] = [0]
        progress_lock = asyncio.Lock()

        try:
            async with asyncio.TaskGroup() as tg:
                for sample in samples:
                    for condition_name, condition in self._config.conditions.items():
                        for repetition_index in range(
                            self._config.execution.num_repetitions
                        ):
                            tg.create_task(
                                self._run_one_triple(
                                    sem=sem,
                                    run_id=run_id,
                                    sample=sample,
                                    condition_name=condition_name,
                                    condition=condition,
                                    repetition_index=repetition_index,
                                    results=results,
                                    total_triples=total_triples,
                                    completed_count=completed_count,
                                    progress_lock=progress_lock,
                                )
                            )
        except* KEvalError as eg:
            # Observer was already called inside _run_one_triple for each failure.
            # Raise the first error as a plain KEvalError to the caller.
            raise eg.exceptions[0]

        # Sort results deterministically: (sample_idx, condition, repetition_index).
        results.sort(
            key=lambda r: (r.sample.sample_idx, r.condition, r.repetition_index)
        )

        self._observer.evaluation_completed(
            run_id=run_id,
            total_runs=len(results),
            elapsed_seconds=time.monotonic() - started_at,
        )

        return RunSummary(
            run_id=run_id,
            dataset_sha256=load_result.sha256,
            config_name=self._config.name,
            runs=results,
        )

    async def _run_one_triple(
        self,
        sem: asyncio.Semaphore,
        run_id: str,
        sample: Sample,
        condition_name: str,
        condition: ConditionConfig,
        repetition_index: int,
        results: list[EvaluationRun],
        total_triples: int,
        completed_count: list[int],
        progress_lock: asyncio.Lock,
    ) -> None:
        """Execute one (sample, condition, repetition_index) triple with retry and backoff.

        The semaphore is held only during active agent/judge calls. The sleep
        between retry attempts happens outside the semaphore so that other tasks
        can proceed during the wait.
        """
        retry_cfg = self._config.execution.retry
        max_attempts = retry_cfg.max_attempts
        backoff = float(retry_cfg.initial_backoff_seconds)

        self._observer.sample_condition_started(
            run_id=run_id,
            sample_idx=sample.sample_idx,
            condition=condition_name,
            repetition_index=repetition_index,
        )

        for attempt in range(1, max_attempts + 1):
            async with sem:
                try:
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
                            repetition_index=repetition_index,
                            agent_result=agent_result,
                            judge_result=judge_result,
                        )
                    )

                    self._observer.sample_condition_completed(
                        run_id=run_id,
                        sample_idx=sample.sample_idx,
                        condition=condition_name,
                        repetition_index=repetition_index,
                    )
                    async with progress_lock:
                        completed_count[0] += 1
                        self._observer.evaluation_progress(
                            run_id=run_id,
                            condition=condition_name,
                            completed=completed_count[0],
                            total=total_triples,
                        )
                    return  # success

                except KEvalError as exc:
                    if not exc.retriable or attempt == max_attempts:
                        self._observer.sample_condition_failed(
                            run_id=run_id,
                            sample_idx=sample.sample_idx,
                            condition=condition_name,
                            repetition_index=repetition_index,
                            reason=str(exc),
                        )
                        async with progress_lock:
                            completed_count[0] += 1
                            self._observer.evaluation_progress(
                                run_id=run_id,
                                condition=condition_name,
                                completed=completed_count[0],
                                total=total_triples,
                            )
                        raise

                    self._observer.sample_condition_retry(
                        run_id=run_id,
                        sample_idx=sample.sample_idx,
                        condition=condition_name,
                        repetition_index=repetition_index,
                        attempt=attempt,
                        reason=str(exc),
                        backoff_seconds=backoff,
                    )
            # Semaphore released here — sleep outside the semaphore block.
            await asyncio.sleep(backoff)
            backoff *= retry_cfg.backoff_multiplier
