"""LiteLLMJudgeFactory — constructs LiteLLMJudge instances."""

import litellm

from judge.domain.judge import Judge
from judge.domain.observer import JudgeObserver
from judge.infrastructure.litellm import LiteLLMJudge
from config.domain.judge import JudgeConfig


class LiteLLMJudgeFactory:
    """Creates LiteLLMJudge instances configured for a given condition and sample."""

    def __init__(self, config: JudgeConfig, observer: JudgeObserver) -> None:
        litellm.suppress_debug_info = True
        self._config = config
        self._observer = observer

    def create(
        self,
        condition: str,
        sample_idx: str,
    ) -> Judge:
        """Construct a new LiteLLMJudge for the given condition and sample."""
        return LiteLLMJudge(
            config=self._config,
            condition=condition,
            sample_idx=sample_idx,
            observer=self._observer,
        )
