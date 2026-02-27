"""AgentResult value object — the outcome of a single agent invocation."""

from pydantic import BaseModel, ConfigDict

from k_eval.agent.domain.turn import AgentTurn
from k_eval.agent.domain.usage import UsageMetrics


class AgentResult(BaseModel, frozen=True):
    """Immutable value object capturing all outcome data from one agent invocation."""

    model_config = ConfigDict(frozen=True)

    response: str
    cost_usd: float | None
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    usage: UsageMetrics | None
    turns: list[AgentTurn] = []
