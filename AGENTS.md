# Project Context for AI Coding Assistants

## Project Overview

**Project Name:** k-eval

**Description:** Context-aware evaluation framework for AI agents using MCP. 

**Tech Stack:**
- Language: Python 3.13
- Key Dependencies: Pydantic, Pydantic Settings, UV, structlog

## Architecture

k-eval strictly follows domain driven design and layered architecture. It 
also strictly follows domain driven observability, making use of domain probes
instead of direct logger calls.

## Coding Guidelines

### Style and Conventions

- Always use type hinting
- Prefer StrEnum, dataclasses, or Pydantic models instead of generic return/parameter types (ex. tuple[], dict[], etc.)
- Use Pydantic settings for all configuration.

### Best Practices

- Always follow TDD, writing failing tests before implementation.
- Keep functions small and focused (single responsibility)
- Add docstrings/comments for complex logic, not obvious code
- Avoid placeholder comments like "next step" or "do the thing"
- When catching exceptions, use specific types, not broad catch-all
- Error messages should start with "Failed to ..."
- Simplicity is good. Repeated code should be extracted to follow DRY principles.

### AI-Specific Guidelines

**Stay Focused:**
- Only modify code directly related to the task
- Avoid "drive-by" improvements or refactoring unrelated code
- If you notice needed changes outside the task scope, note them separately

**Code Quality:**
- Don't repeat yourself (DRY principle)
- Extract boilerplate into reusable functions
- Prefer clear, simple solutions over clever, complex ones

**Testing:**
- New features require tests
- Tests must fail before implementation (TDD)
- Extract test boilerplate into helper functions

## Common Tasks

### Adding a New Feature
1. Review existing similar features for patterns
2. Add tests first (should fail)
3. Implement feature
4. Ensure all tests pass
5. Update documentation

### Fixing a Bug
1. Write a test that reproduces the bug
2. Fix the bug
3. Verify the test passes
4. Check for similar bugs elsewhere

## Important Constraints

- **Security:** Never commit secrets, API keys, or credentials
- **Licensing:** Only use dependencies with compatible licenses
- **Git**: Always use conventional commits. Never skip verify when comitting or pushing. 


## Notes for AI Assistants

- This project uses `uv` for package management. You must use `uv run ...` when running python, pytest, etc.
