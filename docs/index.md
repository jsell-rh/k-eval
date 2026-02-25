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

## Getting Started

See the [Quick Start](../README.md#quick-start) in the README.

## Evaluation Methodology

Please reference [evaluation methodology](./evaluation-methodology.md)
