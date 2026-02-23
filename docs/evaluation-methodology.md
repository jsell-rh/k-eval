# Evaluation Methodology

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

## The Agent

## The Judge


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