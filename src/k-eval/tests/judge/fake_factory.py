"""FakeJudgeFactory — in-memory JudgeFactory implementation for use in tests."""

from judge.domain.judge import Judge
from judge.domain.score import JudgeResult
from tests.judge.fake_judge import FakeJudge


class FakeJudgeFactory:
    """Satisfies the JudgeFactory protocol. Returns a FakeJudge for every create call."""

    def __init__(self, result: JudgeResult | None = None) -> None:
        self._result = result
        self.created: list[dict[str, str]] = []

    def create(
        self,
        condition: str,
        sample_id: str,
    ) -> Judge:
        self.created.append({"condition": condition, "sample_id": sample_id})
        if self._result is not None:
            return FakeJudge(result=self._result)
        return FakeJudge()
