"""EvaluationRun — the result of a single (sample, condition, run_index) evaluation."""

from pydantic import BaseModel, Field

from agent.domain.result import AgentResult
from dataset.domain.sample import Sample
from judge.domain.score import JudgeResult

type RunId = str


class EvaluationRun(BaseModel, frozen=True):
    """Immutable record of one complete evaluation: agent asked, judge scored."""

    run_id: RunId = Field(min_length=1)
    sample: Sample
    condition: str = Field(min_length=1)
    repetition_index: int = Field(ge=0)
    agent_result: AgentResult
    judge_result: JudgeResult
