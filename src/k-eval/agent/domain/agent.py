"""Agent Protocol — structural interface for all agent implementations."""

from typing import Protocol

from agent.domain.result import AgentResult


class Agent(Protocol):
    """Structural interface satisfied by any agent implementation.

    Each instance is constructed once per (condition, sample) evaluation.
    """

    async def ask(self, question: str) -> AgentResult: ...
