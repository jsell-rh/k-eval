"""LiteLLMJudgeFactory — constructs LiteLLMJudge instances."""

import litellm

from k_eval.judge.domain.judge import Judge
from k_eval.judge.domain.observer import JudgeObserver
from k_eval.judge.infrastructure.litellm import LiteLLMJudge
from k_eval.config.domain.judge import JudgeConfig


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
