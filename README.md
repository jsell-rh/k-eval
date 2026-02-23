# k-eval

Context-aware evaluation framework for AI agents using MCP.

## Installation

k-eval uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it first if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From source

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
