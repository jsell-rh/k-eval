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
name: "k-eval-demo"
version: "1"

dataset:
  path: "./questions.jsonl"
  question_key: "question"
  answer_key: "answer"

agent:
  type: "claude_code_sdk"
  model: "claude-sonnet-4-5"

judge:
  # A LiteLLM-compatible provider/model string
  # Note: Credentials should be provided as environment variables.
  # See: https://docs.litellm.ai/docs/providers
  model: "vertex_ai/claude-opus-4-5"
  temperature: 0.0

mcp_servers:
  graph:
    type: "stdio"
    command: "python"
    args: ["-m", "kartograph.mcp"]

  rag:
    type: "sse"
    url: "http://localhost:8080/sse"
    headers:
      Authorization: "Bearer ${SEARCH_API_KEY}"

conditions:
  baseline:
    mcp_servers: []
    system_prompt: |
      Answer the following question using your own knowledge.

  with_graph:
    mcp_servers: [graph]
    system_prompt: |
      You have access to a property graph tool. Use it to answer the question.

  with_graph_and_search:
    mcp_servers: [graph, rag]
    system_prompt: |
      You have access to a property graph tool and a rag tool. 
      Use them to answer the question.
      
execution:
  num_samples: 3
  max_concurrent: 5
  retry:
    max_attempts: 3
    initial_backoff_seconds: 1
    backoff_multiplier: 2
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