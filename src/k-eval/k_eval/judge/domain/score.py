"""JudgeResult — structured output from a single judge invocation."""

from pydantic import BaseModel, ConfigDict, Field


class JudgeResult(BaseModel):
    """Immutable Pydantic model capturing all scored metrics from one judge call.

    Used as a cross-layer DTO: produced by judge infrastructure, consumed by
    application and reporting layers.
    """

    model_config = ConfigDict(frozen=True)

    factual_adherence: int = Field(ge=1, le=5)
    factual_adherence_reasoning: str
    completeness: int = Field(ge=1, le=5)
    completeness_reasoning: str
    helpfulness_and_clarity: int = Field(ge=1, le=5)
    helpfulness_and_clarity_reasoning: str
    unverified_claims: list[str] = Field(default_factory=list)
