"""Tests for LiteLLMJudge infrastructure implementation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from config.domain.judge import JudgeConfig
from judge.infrastructure.errors import JudgeInvocationError
from judge.infrastructure.litellm import LiteLLMJudge
from tests.judge.fake_observer import FakeJudgeObserver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> JudgeConfig:
    return JudgeConfig(model=model, temperature=temperature)


def _make_judge(
    config: JudgeConfig | None = None,
    condition: str = "baseline",
    sample_idx: str = "0",
    observer: FakeJudgeObserver | None = None,
) -> tuple[LiteLLMJudge, FakeJudgeObserver]:
    obs = observer if observer is not None else FakeJudgeObserver()
    cfg = config if config is not None else _make_config()
    judge = LiteLLMJudge(
        config=cfg,
        condition=condition,
        sample_idx=sample_idx,
        observer=obs,
    )
    return judge, obs


def _make_result_json(
    factual_adherence: int = 5,
    factual_adherence_reasoning: str = "Accurate.",
    completeness: int = 5,
    completeness_reasoning: str = "Complete.",
    helpfulness_and_clarity: int = 5,
    helpfulness_and_clarity_reasoning: str = "Clear.",
    unverified_claims: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "factual_adherence": factual_adherence,
            "factual_adherence_reasoning": factual_adherence_reasoning,
            "completeness": completeness,
            "completeness_reasoning": completeness_reasoning,
            "helpfulness_and_clarity": helpfulness_and_clarity,
            "helpfulness_and_clarity_reasoning": helpfulness_and_clarity_reasoning,
            "unverified_claims": unverified_claims
            if unverified_claims is not None
            else [],
        }
    )


def _make_acompletion_response(content: str) -> MagicMock:
    """Build a mock litellm response object with the given message content."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Construction — temperature warning
# ---------------------------------------------------------------------------


class TestConstruction:
    """LiteLLMJudge emits a temperature warning when temperature > 0.0."""

    def test_zero_temperature_emits_no_warning(self) -> None:
        _, observer = _make_judge(config=_make_config(temperature=0.0))

        assert len(observer.temperature_warnings) == 0

    def test_positive_temperature_emits_warning(self) -> None:
        _, observer = _make_judge(config=_make_config(temperature=0.7))

        assert len(observer.temperature_warnings) == 1

    def test_temperature_warning_carries_correct_temperature(self) -> None:
        _, observer = _make_judge(
            config=_make_config(temperature=0.7),
            condition="with-graph",
            sample_idx="3",
        )

        warning = observer.temperature_warnings[0]
        assert warning.temperature == pytest.approx(0.7)
        assert warning.condition == "with-graph"
        assert warning.sample_idx == "3"

    def test_very_small_positive_temperature_emits_warning(self) -> None:
        _, observer = _make_judge(config=_make_config(temperature=0.01))

        assert len(observer.temperature_warnings) == 1


# ---------------------------------------------------------------------------
# score() — success path
# ---------------------------------------------------------------------------


class TestScoreSuccess:
    """score() returns a correct JudgeResult and emits the right events."""

    async def test_returns_parsed_judge_result(self) -> None:
        content = _make_result_json(
            factual_adherence=4,
            factual_adherence_reasoning="Good.",
            completeness=3,
            completeness_reasoning="Missing detail.",
            helpfulness_and_clarity=5,
            helpfulness_and_clarity_reasoning="Clear.",
        )
        mock_response = _make_acompletion_response(content=content)
        judge, _ = _make_judge()

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await judge.score(
                question="What is X?",
                golden_answer="X is 42.",
                agent_response="X is 42.",
            )

        assert result.factual_adherence == 4
        assert result.completeness == 3
        assert result.helpfulness_and_clarity == 5

    async def test_emits_started_event_before_scoring(self) -> None:
        content = _make_result_json()
        mock_response = _make_acompletion_response(content=content)
        judge, observer = _make_judge(condition="with-graph", sample_idx="7")

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            await judge.score(
                question="Q",
                golden_answer="A",
                agent_response="R",
            )

        assert len(observer.started) == 1
        assert observer.started[0].condition == "with-graph"
        assert observer.started[0].sample_idx == "7"
        assert observer.started[0].model == "gpt-4o"

    async def test_emits_completed_event_on_success(self) -> None:
        content = _make_result_json()
        mock_response = _make_acompletion_response(content=content)
        judge, observer = _make_judge(condition="baseline", sample_idx="1")

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            await judge.score(
                question="Q",
                golden_answer="A",
                agent_response="R",
            )

        assert len(observer.completed) == 1
        assert observer.completed[0].condition == "baseline"
        assert observer.completed[0].sample_idx == "1"
        assert observer.completed[0].duration_ms >= 0

    async def test_no_failed_event_on_success(self) -> None:
        content = _make_result_json()
        mock_response = _make_acompletion_response(content=content)
        judge, observer = _make_judge()

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            await judge.score(
                question="Q",
                golden_answer="A",
                agent_response="R",
            )

        assert len(observer.failed) == 0

    async def test_unverified_claims_are_returned(self) -> None:
        content = _make_result_json(
            unverified_claims=["Extra fact A", "Extra fact B"],
        )
        mock_response = _make_acompletion_response(content=content)
        judge, _ = _make_judge()

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await judge.score(
                question="Q",
                golden_answer="A",
                agent_response="R",
            )

        assert result.unverified_claims == ["Extra fact A", "Extra fact B"]


# ---------------------------------------------------------------------------
# score() — litellm exception path
# ---------------------------------------------------------------------------


class TestScoreLiteLLMFailure:
    """score() wraps litellm exceptions as JudgeInvocationError."""

    async def test_litellm_exception_raises_judge_invocation_error(self) -> None:
        judge, _ = _make_judge()

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(side_effect=openai.APIConnectionError(request=MagicMock())),
        ):
            with pytest.raises(JudgeInvocationError):
                await judge.score(
                    question="Q",
                    golden_answer="A",
                    agent_response="R",
                )

    async def test_litellm_exception_message_starts_with_failed(self) -> None:
        judge, _ = _make_judge()

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(side_effect=openai.APIConnectionError(request=MagicMock())),
        ):
            with pytest.raises(JudgeInvocationError) as exc_info:
                await judge.score(
                    question="Q",
                    golden_answer="A",
                    agent_response="R",
                )

        assert str(exc_info.value).startswith("Failed to ")

    async def test_litellm_exception_emits_failed_event(self) -> None:
        judge, observer = _make_judge(condition="with-graph", sample_idx="5")

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(side_effect=openai.APITimeoutError(request=MagicMock())),
        ):
            with pytest.raises(JudgeInvocationError):
                await judge.score(
                    question="Q",
                    golden_answer="A",
                    agent_response="R",
                )

        assert len(observer.failed) == 1
        assert observer.failed[0].condition == "with-graph"
        assert observer.failed[0].sample_idx == "5"

    async def test_litellm_exception_does_not_emit_completed_event(self) -> None:
        judge, observer = _make_judge()

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(side_effect=openai.APITimeoutError(request=MagicMock())),
        ):
            with pytest.raises(JudgeInvocationError):
                await judge.score(
                    question="Q",
                    golden_answer="A",
                    agent_response="R",
                )

        assert len(observer.completed) == 0


# ---------------------------------------------------------------------------
# score() — unparseable response
# ---------------------------------------------------------------------------


class TestScoreParseFailure:
    """score() raises JudgeInvocationError when response cannot be parsed."""

    async def test_invalid_json_raises_judge_invocation_error(self) -> None:
        mock_response = _make_acompletion_response(content="not valid json at all")
        judge, _ = _make_judge()

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            with pytest.raises(JudgeInvocationError):
                await judge.score(
                    question="Q",
                    golden_answer="A",
                    agent_response="R",
                )

    async def test_invalid_json_message_starts_with_failed(self) -> None:
        mock_response = _make_acompletion_response(content="{}")
        judge, _ = _make_judge()

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            with pytest.raises(JudgeInvocationError) as exc_info:
                await judge.score(
                    question="Q",
                    golden_answer="A",
                    agent_response="R",
                )

        assert str(exc_info.value).startswith("Failed to ")

    async def test_invalid_response_emits_failed_event(self) -> None:
        mock_response = _make_acompletion_response(content="garbage")
        judge, observer = _make_judge(condition="baseline", sample_idx="2")

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            with pytest.raises(JudgeInvocationError):
                await judge.score(
                    question="Q",
                    golden_answer="A",
                    agent_response="R",
                )

        assert len(observer.failed) == 1
        assert observer.failed[0].condition == "baseline"
        assert observer.failed[0].sample_idx == "2"

    async def test_invalid_response_does_not_emit_completed_event(self) -> None:
        mock_response = _make_acompletion_response(content="garbage")
        judge, observer = _make_judge()

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            with pytest.raises(JudgeInvocationError):
                await judge.score(
                    question="Q",
                    golden_answer="A",
                    agent_response="R",
                )

        assert len(observer.completed) == 0

    async def test_out_of_range_score_in_response_raises_judge_invocation_error(
        self,
    ) -> None:
        # Valid JSON but JudgeResult validation fails (score=0 is out of range)
        content = json.dumps(
            {
                "factual_adherence": 0,
                "factual_adherence_reasoning": "Bad.",
                "completeness": 3,
                "completeness_reasoning": "Fine.",
                "helpfulness_and_clarity": 3,
                "helpfulness_and_clarity_reasoning": "Fine.",
                "unverified_claims": [],
            }
        )
        mock_response = _make_acompletion_response(content=content)
        judge, _ = _make_judge()

        with patch(
            "judge.infrastructure.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            with pytest.raises(JudgeInvocationError):
                await judge.score(
                    question="Q",
                    golden_answer="A",
                    agent_response="R",
                )
