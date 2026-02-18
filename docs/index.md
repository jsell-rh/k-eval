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


### 1. Accuracy (1-5)

Are the facts included in the response accurate when compared to 
golden response?

| Score | Description |
|-------|-------------|
| 1 | None of the facts included in the response are accurate according to the golden response. |
| 2 | The response contains key factual errors according to the golden response, but includes a few correct details. |
| 3 | The key facts are mostly accurate according to the golden response, but there is an inaccurate key fact or a couple inaccurate details. |
| 4 | All key facts are accurate according to the golden response, but there are minor inaccuracies in details. |
| 5 | All key facts and additional details are accurate according to the golden response. Key commands or phrases are included verbatim. |


### 2. Completeness (1-5)

Does the response contain all of the facts from the golden response?

| Score | Description |
|-------|-------------|
| 1 | The response doesn't include any facts present in the golden response. |
| 2 | The response contains one or two details from the golden response, but misses the primary answer. |
| 3 | The response is mostly helpful, but does not contain all the facts provided in the golden response. |
| 4 | The response contains all the facts from the golden response, but is missing a few non-essential details. |
| 5 | The response contains all the facts and details from the golden response. | 


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
