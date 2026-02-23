"""Tests for AgentFactoryRegistry — create_agent_factory routing."""

import pytest

from agent.infrastructure.errors import AgentTypeNotSupportedError
from agent.infrastructure.factory import ClaudeAgentSDKAgentFactory
from agent.infrastructure.registry import create_agent_factory
from config.domain.agent import AgentConfig
from tests.agent.fake_observer import FakeAgentObserver


def _make_config(agent_type: str = "claude_code_sdk") -> AgentConfig:
    return AgentConfig(type=agent_type, model="claude-3-5-sonnet-20241022")


class TestCreateAgentFactory:
    """create_agent_factory routes to the correct factory based on AgentConfig.type."""

    def test_claude_code_sdk_type_returns_claude_sdk_factory(self) -> None:
        config = _make_config(agent_type="claude_code_sdk")
        observer = FakeAgentObserver()

        factory = create_agent_factory(config=config, observer=observer)

        assert isinstance(factory, ClaudeAgentSDKAgentFactory)

    def test_unknown_type_raises_agent_type_not_supported_error(self) -> None:
        config = _make_config(agent_type="unknown_agent_type")
        observer = FakeAgentObserver()

        with pytest.raises(AgentTypeNotSupportedError):
            create_agent_factory(config=config, observer=observer)

    def test_unknown_type_error_message_starts_with_failed(self) -> None:
        config = _make_config(agent_type="gpt_computer_use")
        observer = FakeAgentObserver()

        with pytest.raises(AgentTypeNotSupportedError) as exc_info:
            create_agent_factory(config=config, observer=observer)

        assert str(exc_info.value).startswith("Failed to ")

    def test_unknown_type_error_message_includes_type_name(self) -> None:
        config = _make_config(agent_type="gpt_computer_use")
        observer = FakeAgentObserver()

        with pytest.raises(AgentTypeNotSupportedError) as exc_info:
            create_agent_factory(config=config, observer=observer)

        assert "gpt_computer_use" in str(exc_info.value)

    def test_agent_type_not_supported_is_keval_error(self) -> None:
        from core.errors import KEvalError

        config = _make_config(agent_type="unsupported")
        observer = FakeAgentObserver()

        with pytest.raises(KEvalError):
            create_agent_factory(config=config, observer=observer)
