# Contributing to k-eval

## Development Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and install

```bash
git clone https://github.com/jsell-rh/k-eval.git
cd k-eval/src/k-eval
uv sync --group dev
```

### Install pre-commit hooks

Two hook stages are used — both must be installed:

```bash
pre-commit install                        # runs on git commit (ruff, mypy)
pre-commit install --hook-type pre-push   # runs on git push (pytest, AGENTS.md validation)
```

Skipping either install means quality checks won't run locally before your code reaches CI.

### Run k-eval locally

```bash
cd src/k-eval
uv run k-eval /path/to/config.yaml
```

### Run the test suite

```bash
cd src/k-eval
uv run pytest
```

## Conventions

- Commits must follow [Conventional Commits](https://www.conventionalcommits.org/) — this drives automatic versioning and the changelog via release-please.
- All PRs must have a conventional commit-style title (enforced by CI).
- Do not skip pre-commit hooks (`--no-verify`).

## Architecture

See [AGENTS.md](AGENTS.md) for the domain-driven design conventions, observability patterns, and coding guidelines used in this project.
