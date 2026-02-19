# k-eval

`k-eval` is a context-aware evaluation framework for AI agents using MCP.


## Motivation 

While building [Kartograph](https://github.com/openshif-hyperfleet/kartograph), we needed a method for 
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

> [!Note]
> We considered other methods of evaluation (such as assessing the semantic similarity of the retrieved content to the user's query, etc.).
> However, `k-eval` is laser focused on assessing the final quality of the response and is entirely uninterested in assessing other components
> of the system.

_The below are nearly-verbatim instructions provided to the LLM as a judge._

### 1. Factual Adherence and Safety (1-5)

Are the facts and commands (if applicable) included in the response accurate and safe according to the golden response? 
If the response includes additional context or steps not present in the golden response,
do they logically align with the solution space of the user's query without contradicting the
golden reference or inventing non-existent commands (if applicable)?

> Note: If you are unsure whether additional information in the response is accurate,
> but it does not contradict the golden data, do not automatically fail it.
> Flag it as an 'unverified claim' but score based on the accuracy of the core golden facts.

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


### 3. Conciseness (1-5)

Does the response present the facts in a consumable way, minimizing
irrelevant "fluff"?

| Score | Description |
|-------|-------------|
| 1 | The response is overwhelmed by irrelevant information, hallucinations, or filler that obscures the answer, if present. |
| 2 | Significant portions of the response are irrelevant to the golden response. |
| 3 | The response is helpful, but contains repetitive or unnecessary prose. |
| 4 | The response is direct, with only minimal lead-in or concluding sentences. |
| 5 | The response is direct, every sentence is a significant contribution to answering the user query according to the golden response. |
