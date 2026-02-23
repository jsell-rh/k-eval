"""Tests verifying AgentFactory protocol compliance of ClaudeAgentSDKAgentFactory."""

import inspect

from agent.domain.factory import AgentFactory
from agent.infrastructure.factory import ClaudeAgentSDKAgentFactory
from tests.agent.fake_factory import FakeAgentFactory
from agent.domain.result import AgentResult
from agent.domain.usage import UsageMetrics


def _make_result() -> AgentResult:
    return AgentResult(
        response="test",
        cost_usd=None,
        duration_ms=100,
        duration_api_ms=90,
        num_turns=1,
        usage=UsageMetrics(input_tokens=10, output_tokens=5),
    )


class TestClaudeAgentSDKAgentFactoryProtocolCompliance:
    """ClaudeAgentSDKAgentFactory satisfies the AgentFactory protocol."""

    def test_factory_has_create_method(self) -> None:
        assert hasattr(ClaudeAgentSDKAgentFactory, "create")
        assert callable(ClaudeAgentSDKAgentFactory.create)

    def test_factory_create_accepts_required_parameters(self) -> None:
        sig = inspect.signature(ClaudeAgentSDKAgentFactory.create)
        params = list(sig.parameters.keys())

        assert "condition" in params
        assert "sample_id" in params
        assert "system_prompt" in params
        assert "mcp_servers" in params


class TestFakeAgentFactoryProtocolCompliance:
    """FakeAgentFactory satisfies the AgentFactory protocol."""

    def test_fake_factory_satisfies_agent_factory_protocol(self) -> None:
        factory: AgentFactory = FakeAgentFactory(result=_make_result())

        assert factory is not None

    def test_fake_factory_has_create_method(self) -> None:
        factory = FakeAgentFactory(result=_make_result())

        assert hasattr(factory, "create")
        assert callable(factory.create)

    def test_fake_factory_create_returns_agent(self) -> None:
        factory = FakeAgentFactory(result=_make_result())

        agent = factory.create(
            condition="baseline",
            sample_id="s1",
            system_prompt="You are helpful.",
            mcp_servers=[],
        )

        assert agent is not None

    def test_fake_factory_records_create_calls(self) -> None:
        factory = FakeAgentFactory(result=_make_result())

        factory.create(
            condition="baseline",
            sample_id="s1",
            system_prompt="You are helpful.",
            mcp_servers=[],
        )
        factory.create(
            condition="with-graph",
            sample_id="s2",
            system_prompt="You have graph context.",
            mcp_servers=[],
        )

        assert len(factory.created) == 2
        assert factory.created[0]["condition"] == "baseline"
        assert factory.created[1]["condition"] == "with-graph"
