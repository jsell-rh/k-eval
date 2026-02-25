# k-eval

Context-aware evaluation framework for AI agents using MCP.

## Quick Start

k-eval uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it first if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install `k-eval`

```bash
git clone https://github.com/jsell-rh/k-eval.git
cd k-eval/src/k-eval

# Core dependencies
uv sync

# With Vertex AI provider support
uv sync --extra vertex_ai

# All provider dependencies
uv sync --extra all
```

### Run `k-eval`

`k-eval` runs are configured using `yaml` configuration files (see [Configuration](#Configuration)).

Once an evaluation is defined in a `yaml` file, you can invoke
`k-eval` like:

```bash
cd src/k-eval
uv run python -m cli.main /path/to/config.yaml
```

### Understanding the Output

Each run produces two files in `./results/` (or wherever you point `--output-dir`):

```
results/
  my-eval_20260225_a1b2c3d4.json           # aggregate scores per condition
  my-eval_20260225_a1b2c3d4.detailed.jsonl # one line per (question, condition) pair
```

**`{name}_{date}_{run_id}.json`** — the summary. One entry per condition with
mean and standard deviation for each of the three metrics across all questions
and repetitions. Use this to compare conditions at a glance.

This file is intended to be mostly compliant with the [Every Eval Ever](https://evalevalai.com/projects/every-eval-ever/) schema.
Notably, `k-eval` does not aggregate the three metrics into a single score.
Thus, the individual metrics are written to `score_details.details`, and
`score_details.score` is left `null`. 

**`{name}_{date}_{run_id}.detailed.jsonl`** — the full record. One JSON object per
`(question, condition)` pair containing the agent's raw responses for every
repetition, per-repetition judge scores and reasoning, unverified claims, and
token usage. Use this if you want to dig into why a condition scored the way it did.

The three metrics are scored 1-5 by the judge model:

| Metric | What it measures |
|---|---|
| `factual_adherence` | Does the response stick to facts in the golden answer? |
| `completeness` | Does it cover all the essential points? |
| `helpfulness_and_clarity` | Is it well-structured and easy to act on? |

See [evaluation-methodology](docs/evaluation-methodology.md) for more details.

### Configuration

A config file defines your dataset, agent, judge, MCP servers, and the conditions you want to compare:

```yaml
name: "my-eval"
version: "1"

dataset:
  # JSONL file with your questions and golden answers
  path: "./questions.jsonl"
  # The name of the key used to reference the question within the JSONL file.
  question_key: "question"
  # They key used to reference the golden "reference" or answer within the JSON file.
  answer_key: "answer"

agent:
  type: "claude_code_sdk" # currently the only supported type
  model: "claude-sonnet-4-5"

judge:
  model: "vertex_ai/claude-opus-4-5" # any LiteLLM-compatible model string (See: https://models.litellm.ai/)
  temperature: 0.0

mcp_servers:
  graph:
    type: "stdio"
    command: "python"
    args: ["-m", "my_mcp_server"]

conditions:
  baseline:
    mcp_servers: []
    system_prompt: |
        Answer using your own knowledge.
  with_graph:
    mcp_servers: [graph]
    system_prompt: |
        Use the graph tool to answer the question.

execution:
  # How many times each (question, condition) pair is evaluated.
  # This is useful for managing variance in agent responses. Standard
  # deviation between scores will be reported if num_repetitions >= 3
  num_repetitions: 3
  # (question, condition, repetition) tuples can be evaluated concurrently
  # to reduce total evaluation time. The upper bound of this number is determined
  # only by the resources on your computer and by the rate limit configuration
  # of the agent and model providers.
  #
  # In practice, numbers even as high as 50 seem to be well tolerated 
  # when using Vertex AI.
  max_concurrent: 5
```

See [docs/run-configuration.md](docs/run-configuration.md) for the full reference including authentication setup.

