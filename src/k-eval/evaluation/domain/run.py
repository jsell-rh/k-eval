"""EvaluationRun — the result of a single (sample, condition, run_index) evaluation."""

from pydantic import BaseModel, ConfigDict

from agent.domain.result import AgentResult
from dataset.domain.sample import Sample
from judge.domain.score import JudgeResult

type RunId = str


class EvaluationRun(BaseModel, frozen=True):
    """Immutable record of one complete evaluation: agent asked, judge scored.

    Pydantic needs arbitrary_types_allowed because Sample is a stdlib frozen dataclass,
    not a Pydantic model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: RunId
    sample: Sample
    condition: str
    run_index: int
    agent_result: AgentResult
    judge_result: JudgeResult
