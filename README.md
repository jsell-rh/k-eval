# k-eval

Context-aware evaluation framework for AI agents using MCP.

## Installation

k-eval uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it first if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then add k-eval to your project:

```bash
# Core install
uv add k-eval

# With Vertex AI provider support
uv add "k-eval[vertex_ai]"

# All provider dependencies
uv add "k-eval[all]"
```

Once installed, run k-eval via uv:

```bash
uv run k-eval run eval.yaml
```
