# Run Configuration

`k-eval` manages its config, or run specifications, using
`yaml` files. These files capture everything needed to **attempt**
a rerun:
- golden dataset path
- inference parameters
- system prompts
- MCP server connection parameters

> [!Important]
>
> Note that run specifications **do not** provide a means
> to _perfectly replicate_ a run. For more information
> as to why perfect replication is a non-goal, see [evaluation methodology](./evaluation-methodology.md).

## Config Example

```yaml
# A short name for this evaluation run. Used in output file names.
name: "k-eval-demo"
# Treat this as a schema version for your own config files — increment it
# when you make meaningful changes so you can tell configs apart later.
version: "1"

dataset:
  path: "./questions.jsonl"   # JSONL file with your questions and golden answers
  question_key: "question"    # key in each JSONL object that holds the question
  answer_key: "answer"        # key that holds the known-correct reference answer

agent:
  type: "claude_code_sdk"     # currently the only supported type
  model: "claude-sonnet-4-5"  # any model string accepted by the Claude Code SDK

judge:
  # Any LiteLLM-compatible provider/model string. See: https://models.litellm.ai/
  # The judge scores each agent response against the golden answer on three metrics.
  # Use a capable model here. Judge quality directly affects result reliability.
  #
  # The model must support structured outputs.
  model: "vertex_ai/claude-opus-4-5"
  # Keep temperature at 0.0 to make judge scores as deterministic as possible.
  temperature: 0.0

mcp_servers:
  # Define all MCP servers here, then reference them by name in conditions below.
  graph:
    # supported: `stdio`, `sse`, `http`
    type: "stdio" # stdio: the server is a local process
    command: "python"
    args: ["-m", "kartograph.mcp"]

  rag:
    type: "http" # http: the server is a remote http endpoint
    url: "http://localhost:8080/mcp"
    # ${ENV_VAR} syntax is supported anywhere in the config. k-eval will
    # substitute values from your environment at load time.
    headers:
      Authorization: "Bearer ${SEARCH_API_KEY}"

conditions:
  # Each condition is an experimental variant. k-eval evaluates every question
  # under every condition, so you can directly compare the effect of each one.
  # The agent only has access to the MCP servers listed here. All others are
  # withheld, including Claude's built-in tools (web search, bash, etc.).
  baseline:
    mcp_servers: [] # no MCP servers, agent uses only its own knowledge
    system_prompt: |
      Answer the following question using your own knowledge.

  with_graph:
    mcp_servers: [graph] # reference server names defined in mcp_servers above
    system_prompt: |
      You have access to a property graph tool. Use it to answer the question.

  with_graph_and_search:
    mcp_servers: [graph, rag]
    system_prompt: |
      You have access to a property graph tool and a rag tool.
      Use them to answer the question.

execution:
  # How many times each (question, condition) pair is evaluated. The mean and
  # standard deviation across repetitions are reported. Use 3 as a minimum;
  # 5+ gives more reliable variance estimates.
  num_repetitions: 5
  # (question, condition, repetition) triples are evaluated concurrently up to
  # this limit. Higher values reduce wall-clock time but increase API load.
  # Values up to 50 seem to be well-tolerated on Vertex AI in practice.
  max_concurrent: 5
  retry:
    max_attempts: 3 # total attempts per triple including the first
    initial_backoff_seconds: 1
    backoff_multiplier: 2 # backoff doubles on each retry: 1s, 2s, 4s, ...
```

## Inference Provider Authentication

As a general rule, `k-eval` relies on the user's environment
to provide authentication details required to run the agent and
llm judge as configured in the evaluation configuration file.

The required environment variables will differ depending on the agent
used, and are likely to differ slightly between the agent and judge.

For now, we will focus on providing instructions for running Claude Agent SDK
using Vertex AI, and running judge models hosted on Vertex AI.


### Agent

All [environment variables required to run the Claude Agent SDK](https://code.claude.com/docs/en/google-vertex-ai)
must be present in the user's environment:

```bash
# Enable Vertex AI integration
export CLAUDE_CODE_USE_VERTEX=1
export CLOUD_ML_REGION=global
export ANTHROPIC_VERTEX_PROJECT_ID=YOUR-PROJECT-ID

# Disable prompt caching to support variance metrics
export DISABLE_PROMPT_CACHING=1

# When CLOUD_ML_REGION=global, override region for unsupported models
export VERTEX_REGION_CLAUDE_3_5_HAIKU=us-east5
```

### Judge

All [environment variables required to use Vertex AI models with LiteLLM](https://docs.litellm.ai/docs/providers/vertex#environment-variables)
must be present in the user's environment. 

Note that LiteLLM's environment variable names may differ slightly compared
to those used to configure agent authentication.

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"
export VERTEXAI_LOCATION="us-central1" # can be any vertex location
export VERTEXAI_PROJECT="my-test-project" # ONLY use if model project is different from service account project
```