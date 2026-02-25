"""Tests verifying JudgeFactory protocol compliance of LiteLLMJudgeFactory."""

import inspect

from k_eval.judge.domain.factory import JudgeFactory
from k_eval.judge.infrastructure.factory import LiteLLMJudgeFactory
from tests.judge.fake_factory import FakeJudgeFactory


class TestLiteLLMJudgeFactoryProtocolCompliance:
    """LiteLLMJudgeFactory satisfies the JudgeFactory protocol."""

    def test_factory_has_create_method(self) -> None:
        assert hasattr(LiteLLMJudgeFactory, "create")
        assert callable(LiteLLMJudgeFactory.create)

    def test_factory_create_accepts_required_parameters(self) -> None:
        sig = inspect.signature(LiteLLMJudgeFactory.create)
        params = list(sig.parameters.keys())

        assert "condition" in params
        assert "sample_idx" in params


class TestFakeJudgeFactoryProtocolCompliance:
    """FakeJudgeFactory satisfies the JudgeFactory protocol."""

    def test_fake_factory_satisfies_judge_factory_protocol(self) -> None:
        factory: JudgeFactory = FakeJudgeFactory()

        assert factory is not None

    def test_fake_factory_has_create_method(self) -> None:
        factory = FakeJudgeFactory()

        assert hasattr(factory, "create")
        assert callable(factory.create)

    def test_fake_factory_create_returns_judge(self) -> None:
        factory = FakeJudgeFactory()

        judge = factory.create(condition="baseline", sample_idx="s1")

        assert judge is not None

    def test_fake_factory_records_create_calls(self) -> None:
        factory = FakeJudgeFactory()

        factory.create(condition="baseline", sample_idx="s1")
        factory.create(condition="with-graph", sample_idx="s2")

        assert len(factory.created) == 2
        assert factory.created[0]["condition"] == "baseline"
        assert factory.created[1]["condition"] == "with-graph"
