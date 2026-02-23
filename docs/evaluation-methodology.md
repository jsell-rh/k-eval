# Evaluation Methodology

## Table of Contents

- [Evaluation Methodology](#evaluation-methodology)
- [Experiment Structure](#experiment-structure)
- [Samples, Conditions, and Runs](#samples-conditions-and-runs)
- [The Agent](#the-agent)
- [The Judge](#the-judge)
- [Metrics](#metrics)
    - [1. Factual Adherence and Safety (1-5)](#1-factual-adherence-and-safety-1-5)
    - [2. Completeness (1-5)](#2-completeness-1-5)
    - [3. Helpfulness and Clarity (1-5)](#3-helpfulness-and-clarity-1-5)
- [Variance Management](#variance-management)
    - [Variable Definition](#variable-definition)
    - [Variance Management Techniques](#variance-management-techniques)

The overarching goal of `k-eval`'s evaluation methodology is to 
strike a balance between reproducibility and the usefulness of
results.

Unfortunately, the nature of evaluating black box agentic systems
(like Claude Code) in combination with [also black box] MCP servers
means that we cannot guarantee full reproducibility.

Instead, we opt to sample agent responses and report
score variance alongside average scores.

We believe that this approach, though not entirely reproducible,
still produces statistically meaningful results despite
the inherent non-determinism of agentic systems.

See [Variance Management](#variance-management) for more details.


## Experiment Structure

A `k-eval` run evaluates an agentic system against a golden dataset, comprised of
high-quality question/answer pairs, under one or more **conditions**.

A condition is a combination of a system prompt and a set of 0 or more MCP servers.

The condition is the unit of comparison within `k-eval`. Within a run, the agentic system
is constant. The experimental variable is the context that is (or is not) provided
to the agent.

Cross-agent comparison is out of scope for a single run. If you wish to compare
agents, separate runs (config files) can be used and results compared externally.

## Samples, Conditions, and Runs

A **sample** is a single question/answer pair from the golden dataset.

A **condition** is a named experimental configuration specifying which MCP servers (if any)
are available to the agent, and what system prompt is used.

A **run** is the full evaluation of every sample under every condition. For example, a run with
50 samples and 3 conditions would consist of 150 evaluations. With the default per-sample sampling
of `N=3` (see [variance management techniques](#variance-management-techniques)), this would result
in 450 total agent calls and 450 LLM judge calls. (Note that sampled scores are reported as 
average score & standard deviation. So the final reported scores would be the aggregate of 150 scores.)

Conditions are explicitly defined by the user. `k-eval` does not automatically enumerate combinations.
Named conditions serve as meaningful labels in the output, and thus some thought should be used
when naming conditions. For example,

- `baseline`
- `with-graph`
- `with-graph-and-rag`

See [run-configuration.md](./run-configuration.md) for more details on how runs
are defined.

## The Agent

The agent (or agentic system) is the target of `k-eval`'s evaluation efforts. `k-eval`
is designed to be agentic-agnostic. An agent can be evaluated by `k-eval` if it can
be wrapped in an interface like:

```python
def query(user_query: str, system_prompt:str, mcp_servers) -> str:
    ...
```

Note that `k-eval` does not, by design, care about the internal reasoning steps,
tool calls, or other implementation details of the agent. This is both for the 
sake of simplicity, and also to maximize compatibility with current and future systems.
This decision is also a direct consequence of the authors' belief that the most
valuable metric [for their use-case] is the final output of the agent, regardless
of how it achieved that response.

The agent is configured once per run and held constant across all conditions.

> [!Note]
>
> The first (and currently only) agentic system to be supported is Claude Code,
> through the Agent SDK.
>
> Community contributions are welcomed to enable support of additional agents.

## The Judge

The judge is an LLM, distinct from the agent being evaluated, that is used
to score the responses from the agent with regard to the golden dataset answer
and according to the [`k-eval` metrics](#metrics).

The judge's inference parameters are pinned (configuration per-run) in
an attempt to limit variance in its scoring.

The judge produces structured JSON output per evaluation that contains
a score for each metric as well as its reasoning for each score. This reasoning
is provided for human assessment if there is reason to question the judge's score.
Additionally, a section for `unverified_claims` contains any details in the agent's response
that the judge believes are helpful, but that it cannot guarantee are factual.


## Metrics

`k-eval` computes the following metrics using an [LLM as a judge](https://en.wikipedia.org/wiki/LLM-as-a-Judge) technique.

> [!Note]
> **Non-Goal**
>
> `k-eval` is focused on evaluating the final response from the LLM. It is a non-goal to 
> evaluate the performance of retrieval mechanisms that may be used behind an MCP server
> used during evaluation. 


> [!Note]
> **On the Suboptimality of Golden Data**
>
> It became clear early on in our research that golden question/answer
> datasets often did not present as "optimally good". Specifically,
> these datasets often provided the _core_ of a "perfect" response,
> but lacked additional context which could improve the quality of 
> the response to a non-subject matter expert.
>
> To this end, our evaluation metrics do _not_ directly penalize
> responses that include information beyond what is provided in the 
> golden response. 
>
> We recognize that this introduces the possibility that hallucinations
> that should be penalized may be scored highly. To gain visibility into this,
> we instruct the judge LLM to output its evaluation in a structured format (JSON)
> that includes the final score for each metric, a brief explanation of its reasoning, and a dedicated
> list of **unverified_claims**.
> 
> This structured output makes it easy for human operators (or automated
> systems) to filter and review evaluations that contain extra, potentially
> incorrect context without blindly failing the evaluation.

_The below are nearly-verbatim instructions provided to the LLM as a judge._

### 1. Factual Adherence and Safety (1-5)

Are the facts and commands (if applicable) included in the response accurate and safe according to the golden response? 
If the response includes additional context or steps not present in the golden response,
do they logically align with the solution space of the user's query without contradicting the
golden reference or inventing non-existent commands (if applicable)?

> Note: If you are unsure whether additional information in the response is accurate,
> but it does not contradict the golden data, do not automatically fail it.
> **Flag it as an 'unverified claim' but score based on the accuracy of the core golden facts.**

| Score | Description |
|-------|-------------|
| 1 | The response directly contradicts the golden response in a way that would cause harm (ex. destructive commands, wrong flags, incorrect step order for critical tasks.) |
| 2 | The response contains major factual errors when compared to the golden response. It may not be explicitly dangerous, but following the response will lead to failure or confusion. If extra information is provided, it is largely inaccurate. |
| 3 | The core of the golden response is accurately represented in the response, but the response includes additional context or commands that are noticeably inaccurate or confusing. |
| 4 | All core components of the golden response are accurately represented in the response. Any additional context is generally safe and/or plausible, though it may contain minor, non-impactful imprecisions. |
| 5 | All core components of the golden response are accurately represented in the response. Any additional context is completely factually accurate, safe, and does not contradict the golden response in any way. |


### 2. Completeness (1-5)

Does the response contain the essential facts (such as commands, parameters, steps)
provided in the golden response? Focus solely on whether the golden information is present.
Do not penalize the response for including additional context as long as the core golden facts are fully represented.

| Score | Description |
|-------|-------------|
| 1 | The response doesn't include any facts, commands (if applicable), or concepts present in the golden response. |
| 2 | The response contains one or two minor details from the golden response, but misses the primary answer or core command required to answer the user's query. |
| 3 | The response provides the primary answer or command but misses critical parameters, flags, or important contextual facts explicitly stated in the golden response. |
| 4 | The response contains the primary answer and almost all key facts from the golden response. It may be missing a minor, non-essential detail. |
| 5 | The response contains all key facts, primary commands (if applicable), and essential details present in the golden response. (Note: a response still is a 5 if it includes all golden facts alongside additional, helpful content.) | 


### 3. Helpfulness and Clarity (1-5)

Is the response presented in a clear, consumable, and actionable way?
Does it use appropriate formatting (e.g., code blocks for commands) 
and avoid burying the primary answer in unnecessary conversational prose?

| Score | Description |
|-------|-------------|
| 1 | The response is a wall of text, lacks formatting, or is overwhelmingly filler/irrelevant information such that the actual answer is obscured. |
| 2 | The response contains the answer, but it is buried deep within paragraphs of unnecessary prose. Formatting may be inconsistent. |
| 3 | The response is helpful and contains the correct answer, but includes repetitive text or minor "fluff". The response may be made more scannable with better formatting. |
| 4 | The response is direct and easy to scan and read. It may contain a brief introduction or conclusion sentence, but the core answer is highly visible. |
| 5 | The response is perfectly optimized to answer the user's query. Formatting is excellent, enabling scanning. Each sentence directly contributes to answering the query or providing critical context. There is no useless "fluff". |


## Variance Management

This section describes the nature of the variables
at play within `k-eval`, and how variance within
them is managed.

### Variable Definition

To manage variance, we first need to understand which
variables can be fully fixed, and which cannot.

#### Fixed Variables (Reproducibility Guaranteed)

These variables are guaranteed to be reproducible via
mechanisms like hashes.

- Input dataset ("Golden dataset")
    - Reason: A SHA256 digest of the input dataset is stored
      alongside the evaluation results.

#### Best-effort Fixed (Reproducibility Not Guaranteed)

Components of these variables may be controllable
across runs, but there is no guarantee of reproducibility.

- LLM Judge Score
    - Reason: Some model providers allow fixing temperature
      and seed, but GPU sharding and floating point imprecision
      can lead to variance in model response.
- MCP Server Response
    - Reason: Any MCP server connected to `k-eval` is not 
      controlled by `k-eval`. Thus, there is no guarantee
      that the data or mechanism behind the MCP server does
      not change over time.

#### Uncontrolled (Structurally Non-deterministic)

These variables are fundamentally unfixable and there is
absolutely no guarantee of reproducibility.

Variables in this category are the primary targets for
variance management techniques, such as sampling.

- Agent Response
    - Reason: Most agent frameworks today do not allow the user
      to specify inference parameters (such as temperature, or seed).
      This introduces run-to-run variance. More critically, the agent
      loop often involves tool calls to [potentially non-deterministic]
      MCP servers. The result of the tool call is used by the agent to
      inform its next steps. The entire process is inherently stochastic.

### Variance Management Techniques

The primary target for variance management is the agent
response. This is not just because the agent response
is likely to contain the most variance, but also because
the agent response is the primary evaluation target of `k-eval`.

Thus, we are most interested in managing, or at least observing,
the variance in agent response.

> [!Note]
>
> As an aside, the variance in the LLM judge score is also
> important to consider. However, we choose to use a 
> a "best effort" method of managing variance for the LLM Judge 
> scores by fixing the inference parameters (something that 
> cannot be done for the agent responses.)

To manage the variance of the agent response, `k-eval` supports
sampling. A run configuration file can specify the number of
responses that should be independently generated for a given
question in the golden dataset. 

By default, each response is sampled three times to balance
statistical confidence with inference cost.

These responses are then independently scored by the LLM judge.
The score from the judge is then averaged for each metric,
and the per-metric variance is recorded.

This enables understanding not only the agent's performance, 
but also its _reliability_ (by analyzing the standard deviation.)
