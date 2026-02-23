"""Tests for YAML config loading infrastructure."""

from pathlib import Path

import pytest

from tests.config.fake_observer import FakeConfigObserver

# Fixtures directory — absolute so tests are location-independent
# __file__ is tests/config/infrastructure/test_yaml_loader.py
# parent.parent.parent is the tests/ directory
FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


def _fixture(name: str) -> Path:
    return FIXTURES / name


class TestValidConfigLoading:
    """A valid YAML config loads correctly with all fields populated."""

    def test_loads_name_and_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")
        from config.infrastructure.yaml_loader import load_config

        observer = FakeConfigObserver()
        cfg = load_config(_fixture("valid_config.yaml"), observer)

        assert cfg.name == "kartograph-graph-context-eval"
        assert cfg.version == "1"

    def test_loads_dataset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")
        from config.infrastructure.yaml_loader import load_config

        observer = FakeConfigObserver()
        cfg = load_config(_fixture("valid_config.yaml"), observer)

        assert cfg.dataset.path == "./questions.jsonl"
        assert cfg.dataset.question_key == "question"
        assert cfg.dataset.answer_key == "answer"

    def test_loads_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")
        from config.infrastructure.yaml_loader import load_config

        observer = FakeConfigObserver()
        cfg = load_config(_fixture("valid_config.yaml"), observer)

        assert cfg.agent.type == "claude_code_sdk"
        assert cfg.agent.model == "claude-sonnet-4-5"

    def test_loads_judge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")
        from config.infrastructure.yaml_loader import load_config

        observer = FakeConfigObserver()
        cfg = load_config(_fixture("valid_config.yaml"), observer)

        assert cfg.judge.model == "vertex_ai/claude-opus-4-5"
        assert cfg.judge.temperature == 0.0

    def test_loads_stdio_mcp_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")
        from config.infrastructure.yaml_loader import load_config
        from config.domain.mcp_server import StdioMcpServer

        observer = FakeConfigObserver()
        cfg = load_config(_fixture("valid_config.yaml"), observer)

        graph = cfg.mcp_servers["graph"]
        assert isinstance(graph, StdioMcpServer)
        assert graph.command == "python"
        assert graph.args == ["-m", "kartograph.mcp"]

    def test_loads_sse_mcp_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")
        from config.infrastructure.yaml_loader import load_config
        from config.domain.mcp_server import SseMcpServer

        observer = FakeConfigObserver()
        cfg = load_config(_fixture("valid_config.yaml"), observer)

        search = cfg.mcp_servers["search"]
        assert isinstance(search, SseMcpServer)
        assert search.url == "http://localhost:8080/sse"

    def test_loads_conditions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")
        from config.infrastructure.yaml_loader import load_config

        observer = FakeConfigObserver()
        cfg = load_config(_fixture("valid_config.yaml"), observer)

        assert "baseline" in cfg.conditions
        assert "with_graph" in cfg.conditions
        assert "with_graph_and_search" in cfg.conditions

        baseline = cfg.conditions["baseline"]
        assert baseline.mcp_servers == []
        assert "your own knowledge" in baseline.system_prompt

    def test_loads_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")
        from config.infrastructure.yaml_loader import load_config

        observer = FakeConfigObserver()
        cfg = load_config(_fixture("valid_config.yaml"), observer)

        assert cfg.execution.num_samples == 3
        assert cfg.execution.max_concurrent == 5
        assert cfg.execution.retry.max_attempts == 3
        assert cfg.execution.retry.initial_backoff_seconds == 1
        assert cfg.execution.retry.backoff_multiplier == 2

    def test_emits_config_loaded_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")
        from config.infrastructure.yaml_loader import load_config

        observer = FakeConfigObserver()
        load_config(_fixture("valid_config.yaml"), observer)

        assert len(observer.loaded) == 1
        assert observer.loaded[0]["name"] == "kartograph-graph-context-eval"
        assert observer.loaded[0]["version"] == "1"


class TestEnvVarInterpolation:
    """${ENV_VAR} interpolation is applied recursively across all string fields."""

    def test_interpolates_header_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "mysecretkey")
        monkeypatch.setenv("CUSTOM_HEADER", "custom-value")
        monkeypatch.setenv("HTTP_API_KEY", "httpkey")
        from config.infrastructure.yaml_loader import load_config
        from config.domain.mcp_server import SseMcpServer

        observer = FakeConfigObserver()
        cfg = load_config(_fixture("env_var_config.yaml"), observer)

        search = cfg.mcp_servers["search"]
        assert isinstance(search, SseMcpServer)
        assert search.headers is not None
        assert search.headers["Authorization"] == "Bearer mysecretkey"
        assert search.headers["X-Custom"] == "custom-value"

    def test_interpolates_http_server_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "mysecretkey")
        monkeypatch.setenv("CUSTOM_HEADER", "custom-value")
        monkeypatch.setenv("HTTP_API_KEY", "httpkey")
        from config.infrastructure.yaml_loader import load_config
        from config.domain.mcp_server import HttpMcpServer

        observer = FakeConfigObserver()
        cfg = load_config(_fixture("env_var_config.yaml"), observer)

        myhttp = cfg.mcp_servers["myhttp"]
        assert isinstance(myhttp, HttpMcpServer)
        assert myhttp.headers is not None
        assert myhttp.headers["Authorization"] == "Bearer httpkey"

    def test_missing_env_vars_collected_into_single_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure variables are not set
        monkeypatch.delenv("SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_HEADER", raising=False)
        monkeypatch.delenv("HTTP_API_KEY", raising=False)
        from config.infrastructure.yaml_loader import load_config
        from config.infrastructure.errors import MissingEnvVarsError

        observer = FakeConfigObserver()
        with pytest.raises(MissingEnvVarsError) as exc_info:
            load_config(_fixture("env_var_config.yaml"), observer)

        error = exc_info.value
        # All three missing vars must be listed in one error
        assert "SEARCH_API_KEY" in error.missing_vars
        assert "CUSTOM_HEADER" in error.missing_vars
        assert "HTTP_API_KEY" in error.missing_vars

    def test_missing_env_var_error_message_starts_with_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_HEADER", raising=False)
        monkeypatch.delenv("HTTP_API_KEY", raising=False)
        from config.infrastructure.yaml_loader import load_config
        from config.infrastructure.errors import MissingEnvVarsError

        observer = FakeConfigObserver()
        with pytest.raises(MissingEnvVarsError) as exc_info:
            load_config(_fixture("env_var_config.yaml"), observer)

        assert str(exc_info.value).startswith("Failed to ")


class TestConditionValidation:
    """Conditions that reference non-existent MCP servers must fail at load time."""

    def test_unknown_mcp_server_ref_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from config.infrastructure.yaml_loader import load_config
        from config.infrastructure.errors import ConfigValidationError

        observer = FakeConfigObserver()
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(_fixture("invalid_condition_ref.yaml"), observer)

        assert "nonexistent_server" in str(exc_info.value)

    def test_unknown_mcp_server_error_message_starts_with_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from config.infrastructure.yaml_loader import load_config
        from config.infrastructure.errors import ConfigValidationError

        observer = FakeConfigObserver()
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(_fixture("invalid_condition_ref.yaml"), observer)

        assert str(exc_info.value).startswith("Failed to ")


class TestJudgeTemperatureWarning:
    """Judge temperature > 0.0 triggers an observer warning."""

    def test_high_judge_temperature_emits_warning(self) -> None:
        from config.infrastructure.yaml_loader import load_config

        observer = FakeConfigObserver()
        load_config(_fixture("high_temp_judge.yaml"), observer)

        assert len(observer.warnings) == 1
        assert float(observer.warnings[0]["temperature"]) == 0.5

    def test_zero_judge_temperature_no_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")
        from config.infrastructure.yaml_loader import load_config

        observer = FakeConfigObserver()
        load_config(_fixture("valid_config.yaml"), observer)

        assert len(observer.warnings) == 0


class TestConditionOrdering:
    """Condition insertion order must be preserved."""

    def test_conditions_preserve_insertion_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")
        from config.infrastructure.yaml_loader import load_config

        observer = FakeConfigObserver()
        cfg = load_config(_fixture("valid_config.yaml"), observer)

        keys = list(cfg.conditions.keys())
        assert keys == ["baseline", "with_graph", "with_graph_and_search"]
