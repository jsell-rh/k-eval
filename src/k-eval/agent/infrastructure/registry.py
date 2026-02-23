"""AgentFactoryRegistry — maps AgentConfig.type to the correct AgentFactory."""

from agent.domain.factory import AgentFactory
from agent.domain.observer import AgentObserver
from agent.infrastructure.errors import AgentTypeNotSupportedError
from agent.infrastructure.factory import ClaudeAgentSDKAgentFactory
from config.domain.agent import AgentConfig

_SUPPORTED_TYPE = "claude_code_sdk"


def create_agent_factory(config: AgentConfig, observer: AgentObserver) -> AgentFactory:
    """Return the appropriate AgentFactory for the given AgentConfig.

    Raises:
        AgentTypeNotSupportedError: if config.type is not a known agent type.
    """
    if config.type == _SUPPORTED_TYPE:
        return ClaudeAgentSDKAgentFactory(config=config, observer=observer)

    raise AgentTypeNotSupportedError(agent_type=config.type)
