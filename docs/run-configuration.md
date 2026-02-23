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