# k-eval

`k-eval` is a context-aware evaluation framework for AI agents using MCP.


## Motivation 

While building [Kartograph](https://github.com/openshift-hyperfleet/kartograph), we needed a method for 
evaluating an AI agent's performance answering a set of questions with and without access to a property graph.

We found that existing solutions, like [Ragas](https://docs.ragas.io/en/stable/), provided more functionality 
than we needed. While we could have built an evaluation solution around such frameworks, we wanted a solution 
that was tailor-made for our use case.

Our goal is to make `k-eval` extremely low-friction to use. No custom code required to run an evaluation,
just a configuration file. Configuration files should be artifacts that can be used to reproduce prior evaluations.


## Purpose

`k-eval` intends to be a ~~small~~ medium, sharp, tool for evaluating the
performance of AI agents on domain-specific tasks. Specifically, it focuses on evaluating the 
effect of providing different contexts to agents by fixing all variables except
which MCP servers are provided to the agent.

Put more simply, `k-eval` asks an AI agent a set of domain-specific questions (with known-correct answers).
The agent will be provided with 0 or more MCP servers that it can use to fetch context that may be helpful
to answering the question(s). `k-eval` assesses the agent's performance with/without access to additional context,
enabling measuring the direct impact of a MCP-based context provider on agent performance.

## Metrics

`k-eval` computes the following metrics using an [LLM as a judge](https://en.wikipedia.org/wiki/LLM-as-a-Judge) technique.

> [!Note] **Non-Goal**
>
> `k-eval` is focused on evaluating the final response from the LLM. It is a non-goal to 
> evaluate the performance of retrieval mechanisms that may be used behind an MCP server
> used during evaluation. 


> [!Note] **On the Suboptimality of Golden Data**
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
