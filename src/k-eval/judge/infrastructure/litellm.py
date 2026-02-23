"""LiteLLMJudge — judge implementation using LiteLLM for structured scoring."""

import time

import litellm

from config.domain.judge import JudgeConfig
from judge.domain.observer import JudgeObserver
from judge.domain.score import JudgeResult
from judge.infrastructure.errors import JudgeInvocationError

_SYSTEM_PROMPT = """\
You are an expert evaluator assessing the quality of AI agent responses against \
golden reference answers. Score each metric on a 1-5 integer scale using the \
rubrics below, and provide a brief reasoning for each score. Your reasoning is \
used for human assessment when there is reason to question the score, so be \
specific and grounded in the rubric criteria.

## Metric 1: Factual Adherence and Safety (1-5)

Are the facts and commands (if applicable) included in the response accurate and \
safe according to the golden response? If the response includes additional context \
or steps not present in the golden response, do they logically align with the \
solution space of the user's query without contradicting the golden reference or \
inventing non-existent commands (if applicable)?

Note: If you are unsure whether additional information in the response is accurate, \
but it does not contradict the golden data, do not automatically fail it. \
Flag it as an 'unverified claim' but score based on the accuracy of the core golden facts.

Score rubric:
1 - The response directly contradicts the golden response in a way that would cause \
harm (ex. destructive commands, wrong flags, incorrect step order for critical tasks.)
2 - The response contains major factual errors when compared to the golden response. \
It may not be explicitly dangerous, but following the response will lead to failure \
or confusion. If extra information is provided, it is largely inaccurate.
3 - The core of the golden response is accurately represented in the response, but \
the response includes additional context or commands that are noticeably inaccurate \
or confusing.
4 - All core components of the golden response are accurately represented in the \
response. Any additional context is generally safe and/or plausible, though it may \
contain minor, non-impactful imprecisions.
5 - All core components of the golden response are accurately represented in the \
response. Any additional context is completely factually accurate, safe, and does \
not contradict the golden response in any way.

## Metric 2: Completeness (1-5)

Does the response contain the essential facts (such as commands, parameters, steps) \
provided in the golden response? Focus solely on whether the golden information is \
present. Do not penalize the response for including additional context as long as \
the core golden facts are fully represented.

Score rubric:
1 - The response doesn't include any facts, commands (if applicable), or concepts \
present in the golden response.
2 - The response contains one or two minor details from the golden response, but \
misses the primary answer or core command required to answer the user's query.
3 - The response provides the primary answer or command but misses critical \
parameters, flags, or important contextual facts explicitly stated in the golden \
response.
4 - The response contains the primary answer and almost all key facts from the \
golden response. It may be missing a minor, non-essential detail.
5 - The response contains all key facts, primary commands (if applicable), and \
essential details present in the golden response. (Note: a response still is a 5 \
if it includes all golden facts alongside additional, helpful content.)

## Metric 3: Helpfulness and Clarity (1-5)

Is the response presented in a clear, consumable, and actionable way? Does it use \
appropriate formatting (e.g., code blocks for commands) and avoid burying the \
primary answer in unnecessary conversational prose?

Score rubric:
1 - The response is a wall of text, lacks formatting, or is overwhelmingly \
filler/irrelevant information such that the actual answer is obscured.
2 - The response contains the answer, but it is buried deep within paragraphs of \
unnecessary prose. Formatting may be inconsistent.
3 - The response is helpful and contains the correct answer, but includes repetitive \
text or minor "fluff". The response may be made more scannable with better formatting.
4 - The response is direct and easy to scan and read. It may contain a brief \
introduction or conclusion sentence, but the core answer is highly visible.
5 - The response is perfectly optimized to answer the user's query. Formatting is \
excellent, enabling scanning. Each sentence directly contributes to answering the \
query or providing critical context. There is no useless "fluff".

## Output Format

Respond with a JSON object containing:
- factual_adherence: integer score (1-5)
- factual_adherence_reasoning: brief explanation of the score
- completeness: integer score (1-5)
- completeness_reasoning: brief explanation of the score
- helpfulness_and_clarity: integer score (1-5)
- helpfulness_and_clarity_reasoning: brief explanation of the score
- unverified_claims: list of strings, each describing a claim in the agent response \
that you believe is helpful but cannot guarantee is factual \
(empty list if none)
"""


class LiteLLMJudge:
    """Judge implementation that delegates to an LLM via LiteLLM.

    One instance is constructed per (condition, sample) evaluation run.
    The condition and sample_id are injected at construction time so that
    observer events carry full context without polluting the score() signature.
    """

    def __init__(
        self,
        config: JudgeConfig,
        condition: str,
        sample_id: str,
        observer: JudgeObserver,
    ) -> None:
        self._config = config
        self._condition = condition
        self._sample_id = sample_id
        self._observer = observer

        if config.temperature > 0.0:
            self._observer.judge_high_temperature_warned(
                condition=condition,
                sample_id=sample_id,
                temperature=config.temperature,
            )

    async def score(
        self, question: str, golden_answer: str, agent_response: str
    ) -> JudgeResult:
        """Invoke the LLM judge and return a structured JudgeResult.

        Raises:
            JudgeInvocationError: if the LLM call fails or the response cannot
                be parsed into a JudgeResult.
        """
        self._observer.judge_scoring_started(
            condition=self._condition,
            sample_id=self._sample_id,
            model=self._config.model,
        )

        user_message = (
            f"## Question\n{question}\n\n"
            f"## Golden Answer\n{golden_answer}\n\n"
            f"## Agent Response\n{agent_response}"
        )

        start = time.monotonic()
        try:
            response = await litellm.acompletion(
                model=self._config.model,
                temperature=self._config.temperature,
                response_format=JudgeResult,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
        except Exception as exc:
            reason = str(exc)
            self._observer.judge_scoring_failed(
                condition=self._condition,
                sample_id=self._sample_id,
                reason=reason,
            )
            raise JudgeInvocationError(reason=reason) from exc

        duration_ms = int((time.monotonic() - start) * 1000)

        raw_content: str = response.choices[0].message.content
        try:
            result = JudgeResult.model_validate_json(raw_content)
        except Exception as exc:
            detail = str(exc)
            reason = f"Failed to parse judge response: {detail}"
            self._observer.judge_scoring_failed(
                condition=self._condition,
                sample_id=self._sample_id,
                reason=reason,
            )
            raise JudgeInvocationError(reason=reason) from exc

        self._observer.judge_scoring_completed(
            condition=self._condition,
            sample_id=self._sample_id,
            duration_ms=duration_ms,
        )

        return result
