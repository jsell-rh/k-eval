"""AgentFactoryRegistry — maps AgentConfig.type to the correct AgentFactory."""

from k_eval.agent.domain.factory import AgentFactory
from k_eval.agent.domain.observer import AgentObserver
from k_eval.agent.infrastructure.errors import AgentTypeNotSupportedError
from k_eval.agent.infrastructure.factory import ClaudeAgentSDKAgentFactory
from k_eval.config.domain.agent import AgentConfig

_SUPPORTED_TYPE = "claude_code_sdk"


def create_agent_factory(config: AgentConfig, observer: AgentObserver) -> AgentFactory:
    """Return the appropriate AgentFactory for the given AgentConfig.

    Raises:
        AgentTypeNotSupportedError: if config.type is not a known agent type.
    """
    if config.type == _SUPPORTED_TYPE:
        return ClaudeAgentSDKAgentFactory(config=config, observer=observer)

    raise AgentTypeNotSupportedError(agent_type=config.type)
