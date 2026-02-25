"""Tests for JudgeResult domain model."""

import pytest
from pydantic import ValidationError

from k_eval.judge.domain.score import JudgeResult


class TestJudgeResultConstruction:
    """JudgeResult accepts valid scores and rejects out-of-range values."""

    def test_valid_construction_succeeds(self) -> None:
        result = JudgeResult(
            factual_adherence=4,
            factual_adherence_reasoning="Mostly accurate.",
            completeness=3,
            completeness_reasoning="Missing minor detail.",
            helpfulness_and_clarity=5,
            helpfulness_and_clarity_reasoning="Perfectly clear.",
            unverified_claims=["Extra claim not in golden."],
        )

        assert result.factual_adherence == 4
        assert result.completeness == 3
        assert result.helpfulness_and_clarity == 5
        assert result.unverified_claims == ["Extra claim not in golden."]

    def test_score_below_1_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            JudgeResult(
                factual_adherence=0,
                factual_adherence_reasoning="Too low.",
                completeness=3,
                completeness_reasoning="Fine.",
                helpfulness_and_clarity=3,
                helpfulness_and_clarity_reasoning="Fine.",
            )

    def test_score_above_5_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            JudgeResult(
                factual_adherence=3,
                factual_adherence_reasoning="Fine.",
                completeness=6,
                completeness_reasoning="Too high.",
                helpfulness_and_clarity=3,
                helpfulness_and_clarity_reasoning="Fine.",
            )

    def test_all_scores_below_1_raise_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            JudgeResult(
                factual_adherence=0,
                factual_adherence_reasoning="Bad.",
                completeness=0,
                completeness_reasoning="Bad.",
                helpfulness_and_clarity=0,
                helpfulness_and_clarity_reasoning="Bad.",
            )

    def test_all_scores_above_5_raise_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            JudgeResult(
                factual_adherence=6,
                factual_adherence_reasoning="Bad.",
                completeness=6,
                completeness_reasoning="Bad.",
                helpfulness_and_clarity=6,
                helpfulness_and_clarity_reasoning="Bad.",
            )

    def test_helpfulness_and_clarity_below_1_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            JudgeResult(
                factual_adherence=3,
                factual_adherence_reasoning="Fine.",
                completeness=3,
                completeness_reasoning="Fine.",
                helpfulness_and_clarity=0,
                helpfulness_and_clarity_reasoning="Too low.",
            )

    def test_boundary_scores_of_1_and_5_are_valid(self) -> None:
        result = JudgeResult(
            factual_adherence=1,
            factual_adherence_reasoning="Minimum.",
            completeness=5,
            completeness_reasoning="Maximum.",
            helpfulness_and_clarity=3,
            helpfulness_and_clarity_reasoning="Middle.",
        )

        assert result.factual_adherence == 1
        assert result.completeness == 5


class TestJudgeResultDefaults:
    """JudgeResult default values behave correctly."""

    def test_unverified_claims_defaults_to_empty_list(self) -> None:
        result = JudgeResult(
            factual_adherence=5,
            factual_adherence_reasoning="Accurate.",
            completeness=5,
            completeness_reasoning="Complete.",
            helpfulness_and_clarity=5,
            helpfulness_and_clarity_reasoning="Clear.",
        )

        assert result.unverified_claims == []

    def test_unverified_claims_default_is_independent_across_instances(self) -> None:
        result_a = JudgeResult(
            factual_adherence=5,
            factual_adherence_reasoning="A.",
            completeness=5,
            completeness_reasoning="A.",
            helpfulness_and_clarity=5,
            helpfulness_and_clarity_reasoning="A.",
        )
        result_b = JudgeResult(
            factual_adherence=4,
            factual_adherence_reasoning="B.",
            completeness=4,
            completeness_reasoning="B.",
            helpfulness_and_clarity=4,
            helpfulness_and_clarity_reasoning="B.",
        )

        assert result_a.unverified_claims is not result_b.unverified_claims


class TestJudgeResultImmutability:
    """JudgeResult is frozen and cannot be mutated."""

    def test_assigning_field_raises_validation_error(self) -> None:
        result = JudgeResult(
            factual_adherence=3,
            factual_adherence_reasoning="Fine.",
            completeness=3,
            completeness_reasoning="Fine.",
            helpfulness_and_clarity=3,
            helpfulness_and_clarity_reasoning="Fine.",
        )

        with pytest.raises(ValidationError):
            result.factual_adherence = 5  # noqa: E501 — runtime raises ValidationError; mypy does not flag frozen Pydantic assignment
