"""Tests verifying Agent protocol compliance of concrete implementations."""

import inspect

from k_eval.agent.domain.result import AgentResult
from k_eval.agent.domain.usage import UsageMetrics
from k_eval.agent.infrastructure.claude_sdk import ClaudeAgentSDKAgent
from tests.agent.fake_agent import FakeAgent


def _make_result() -> AgentResult:
    return AgentResult(
        response="test response",
        cost_usd=0.001,
        duration_ms=1000,
        duration_api_ms=900,
        num_turns=1,
        usage=UsageMetrics(input_tokens=100, output_tokens=50),
    )


class TestFakeAgentProtocolCompliance:
    """FakeAgent satisfies the Agent protocol."""

    def test_fake_agent_has_ask_method(self) -> None:
        agent = FakeAgent(result=_make_result())

        assert hasattr(agent, "ask")
        assert callable(agent.ask)

    def test_fake_agent_ask_is_coroutine_function(self) -> None:
        agent = FakeAgent(result=_make_result())

        assert inspect.iscoroutinefunction(agent.ask)

    def test_fake_agent_ask_accepts_question_parameter(self) -> None:
        sig = inspect.signature(FakeAgent.ask)
        params = list(sig.parameters.keys())

        assert "question" in params


class TestClaudeAgentSDKAgentProtocolCompliance:
    """ClaudeAgentSDKAgent satisfies the Agent protocol."""

    def test_claude_sdk_agent_has_ask_method(self) -> None:
        assert hasattr(ClaudeAgentSDKAgent, "ask")
        assert callable(ClaudeAgentSDKAgent.ask)

    def test_claude_sdk_agent_ask_is_coroutine_function(self) -> None:
        assert inspect.iscoroutinefunction(ClaudeAgentSDKAgent.ask)

    def test_claude_sdk_agent_ask_accepts_question_parameter(self) -> None:
        sig = inspect.signature(ClaudeAgentSDKAgent.ask)
        params = list(sig.parameters.keys())

        assert "question" in params
