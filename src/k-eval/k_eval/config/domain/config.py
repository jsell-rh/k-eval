"""Top-level EvalConfig aggregate — the root configuration object."""

from pydantic import BaseModel, Field

from k_eval.config.domain.agent import AgentConfig
from k_eval.config.domain.condition import ConditionConfig
from k_eval.config.domain.dataset import DatasetConfig
from k_eval.config.domain.execution import ExecutionConfig
from k_eval.config.domain.judge import JudgeConfig
from k_eval.config.domain.mcp_server import McpServer

type ConditionName = str
type ServerName = str


class EvalConfig(BaseModel, frozen=True):
    """Root configuration aggregate for a k-eval evaluation run."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    dataset: DatasetConfig
    agent: AgentConfig
    judge: JudgeConfig
    mcp_servers: dict[ServerName, McpServer]
    conditions: dict[ConditionName, ConditionConfig] = Field(min_length=1)
    execution: ExecutionConfig
