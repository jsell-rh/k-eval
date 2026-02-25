"""RunSummary — the aggregate result of a completed evaluation run."""

from pydantic import BaseModel, Field

from k_eval.evaluation.domain.run import EvaluationRun


class RunSummary(BaseModel, frozen=True):
    """Immutable summary returned when an evaluation run completes.

    Captures the run identity, the dataset integrity hash used, the evaluation
    configuration name, and every individual EvaluationRun that was executed.
    """

    run_id: str = Field(min_length=1)
    dataset_sha256: str = Field(min_length=1)
    config_name: str = Field(min_length=1)
    runs: list[EvaluationRun]
