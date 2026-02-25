"""Tests for YAML config loading infrastructure."""

from pathlib import Path

import pytest

from k_eval.config.domain.condition_mcp_server import ConditionMcpServer
from k_eval.config.domain.mcp_server import HttpMcpServer, SseMcpServer, StdioMcpServer
from k_eval.config.infrastructure.errors import (
    ConfigLoadError,
    ConfigParseError,
    ConfigValidationError,
    MissingEnvVarsError,
)
from k_eval.config.infrastructure.yaml_loader import YamlConfigLoader
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

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        assert cfg.name == "kartograph-graph-context-eval"
        assert cfg.version == "1"

    def test_loads_dataset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        assert cfg.dataset.path == Path("./questions.jsonl")
        assert cfg.dataset.question_key == "question"
        assert cfg.dataset.answer_key == "answer"

    def test_loads_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        assert cfg.agent.type == "claude_code_sdk"
        assert cfg.agent.model == "claude-sonnet-4-5"

    def test_loads_judge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        assert cfg.judge.model == "vertex_ai/claude-opus-4-5"
        assert cfg.judge.temperature == 0.0

    def test_loads_stdio_mcp_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        graph = cfg.mcp_servers["graph"]
        assert isinstance(graph, StdioMcpServer)
        assert graph.command == "python"
        assert graph.args == ["-m", "kartograph.mcp"]

    def test_loads_sse_mcp_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        search = cfg.mcp_servers["search"]
        assert isinstance(search, SseMcpServer)
        assert search.url == "http://localhost:8080/sse"

    def test_loads_conditions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        assert "baseline" in cfg.conditions
        assert "with_graph" in cfg.conditions
        assert "with_graph_and_search" in cfg.conditions

        baseline = cfg.conditions["baseline"]
        assert baseline.mcp_servers == []
        assert "your own knowledge" in baseline.system_prompt

    def test_loads_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        assert cfg.execution.num_repetitions == 3
        assert cfg.execution.max_concurrent == 5
        assert cfg.execution.retry.max_attempts == 3
        assert cfg.execution.retry.initial_backoff_seconds == 1
        assert cfg.execution.retry.backoff_multiplier == 2

    def test_emits_config_loaded_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        YamlConfigLoader(observer=observer).load(path=_fixture("valid_config.yaml"))

        assert len(observer.loaded) == 1
        assert observer.loaded[0]["name"] == "kartograph-graph-context-eval"
        assert observer.loaded[0]["version"] == "1"


class TestConditionMcpServerResolution:
    """Condition mcp_servers are resolved from names to ConditionMcpServer objects."""

    def test_condition_mcp_servers_is_list_of_condition_mcp_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        with_graph = cfg.conditions["with_graph"]
        assert len(with_graph.mcp_servers) == 1
        assert isinstance(with_graph.mcp_servers[0], ConditionMcpServer)

    def test_condition_mcp_server_name_matches_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        with_graph = cfg.conditions["with_graph"]
        assert with_graph.mcp_servers[0].name == "graph"

    def test_condition_mcp_server_config_matches_top_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        with_graph = cfg.conditions["with_graph"]
        server_config = with_graph.mcp_servers[0].config
        assert isinstance(server_config, StdioMcpServer)
        assert server_config.command == "python"
        assert server_config.args == ["-m", "kartograph.mcp"]

    def test_condition_with_two_servers_resolves_both(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        combined = cfg.conditions["with_graph_and_search"]
        assert len(combined.mcp_servers) == 2
        names = [s.name for s in combined.mcp_servers]
        assert names == ["graph", "search"]

        graph_cfg = combined.mcp_servers[0].config
        search_cfg = combined.mcp_servers[1].config
        assert isinstance(graph_cfg, StdioMcpServer)
        assert isinstance(search_cfg, SseMcpServer)

    def test_condition_with_empty_mcp_servers_resolves_to_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        baseline = cfg.conditions["baseline"]
        assert baseline.mcp_servers == []

    def test_all_invalid_refs_collected_before_raising(self) -> None:
        """All invalid server references across all conditions are reported at once."""
        observer = FakeConfigObserver()
        with pytest.raises(ConfigValidationError) as exc_info:
            YamlConfigLoader(observer=observer).load(
                path=_fixture("invalid_condition_ref.yaml")
            )

        assert "nonexistent_server" in str(exc_info.value)


class TestEnvVarInterpolation:
    """${ENV_VAR} interpolation is applied recursively across all string fields."""

    def test_interpolates_header_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "mysecretkey")
        monkeypatch.setenv("CUSTOM_HEADER", "custom-value")
        monkeypatch.setenv("HTTP_API_KEY", "httpkey")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("env_var_config.yaml")
        )

        search = cfg.mcp_servers["search"]
        assert isinstance(search, SseMcpServer)
        assert search.headers["Authorization"] == "Bearer mysecretkey"
        assert search.headers["X-Custom"] == "custom-value"

    def test_interpolates_http_server_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "mysecretkey")
        monkeypatch.setenv("CUSTOM_HEADER", "custom-value")
        monkeypatch.setenv("HTTP_API_KEY", "httpkey")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("env_var_config.yaml")
        )

        myhttp = cfg.mcp_servers["myhttp"]
        assert isinstance(myhttp, HttpMcpServer)
        assert myhttp.headers["Authorization"] == "Bearer httpkey"

    def test_missing_env_vars_collected_into_single_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure variables are not set
        monkeypatch.delenv("SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_HEADER", raising=False)
        monkeypatch.delenv("HTTP_API_KEY", raising=False)

        observer = FakeConfigObserver()
        with pytest.raises(MissingEnvVarsError) as exc_info:
            YamlConfigLoader(observer=observer).load(
                path=_fixture("env_var_config.yaml")
            )

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

        observer = FakeConfigObserver()
        with pytest.raises(MissingEnvVarsError) as exc_info:
            YamlConfigLoader(observer=observer).load(
                path=_fixture("env_var_config.yaml")
            )

        assert str(exc_info.value).startswith("Failed to ")


class TestConditionValidation:
    """Conditions that reference non-existent MCP servers must fail at load time."""

    def test_unknown_mcp_server_ref_raises_validation_error(self) -> None:
        observer = FakeConfigObserver()
        with pytest.raises(ConfigValidationError) as exc_info:
            YamlConfigLoader(observer=observer).load(
                path=_fixture("invalid_condition_ref.yaml")
            )

        assert "nonexistent_server" in str(exc_info.value)

    def test_unknown_mcp_server_error_message_starts_with_failed(self) -> None:
        observer = FakeConfigObserver()
        with pytest.raises(ConfigValidationError) as exc_info:
            YamlConfigLoader(observer=observer).load(
                path=_fixture("invalid_condition_ref.yaml")
            )

        assert str(exc_info.value).startswith("Failed to ")


class TestJudgeTemperatureWarning:
    """Judge temperature > 0.0 triggers an observer warning."""

    def test_high_judge_temperature_emits_warning(self) -> None:
        observer = FakeConfigObserver()
        YamlConfigLoader(observer=observer).load(path=_fixture("high_temp_judge.yaml"))

        assert len(observer.warnings) == 1
        assert float(observer.warnings[0]["temperature"]) == 0.5

    def test_zero_judge_temperature_no_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        YamlConfigLoader(observer=observer).load(path=_fixture("valid_config.yaml"))

        assert len(observer.warnings) == 0


class TestConditionOrdering:
    """Condition insertion order must be preserved."""

    def test_conditions_preserve_insertion_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API_KEY", "secret123")

        observer = FakeConfigObserver()
        cfg = YamlConfigLoader(observer=observer).load(
            path=_fixture("valid_config.yaml")
        )

        keys = list(cfg.conditions.keys())
        assert keys == ["baseline", "with_graph", "with_graph_and_search"]


class TestMissingConfigFile:
    """Loading a config file that does not exist raises ConfigLoadError."""

    def test_missing_file_raises_config_load_error(self) -> None:
        observer = FakeConfigObserver()
        with pytest.raises(ConfigLoadError) as exc_info:
            YamlConfigLoader(observer=observer).load(
                path=Path("/nonexistent/path/config.yaml")
            )

        assert str(exc_info.value).startswith("Failed to ")

    def test_missing_file_error_includes_path(self) -> None:
        missing = Path("/nonexistent/path/config.yaml")
        observer = FakeConfigObserver()
        with pytest.raises(ConfigLoadError) as exc_info:
            YamlConfigLoader(observer=observer).load(path=missing)

        assert str(missing) in str(exc_info.value)


class TestInvalidYamlFile:
    """Loading a file with invalid YAML syntax raises ConfigParseError."""

    def test_invalid_yaml_raises_config_parse_error(self) -> None:
        observer = FakeConfigObserver()
        with pytest.raises(ConfigParseError):
            YamlConfigLoader(observer=observer).load(path=_fixture("invalid_yaml.yaml"))

    def test_invalid_yaml_error_message_starts_with_failed(self) -> None:
        observer = FakeConfigObserver()
        with pytest.raises(ConfigParseError) as exc_info:
            YamlConfigLoader(observer=observer).load(path=_fixture("invalid_yaml.yaml"))

        assert str(exc_info.value).startswith("Failed to ")

    def test_invalid_yaml_is_keval_error(self) -> None:
        from k_eval.core.errors import KEvalError

        observer = FakeConfigObserver()
        with pytest.raises(KEvalError):
            YamlConfigLoader(observer=observer).load(path=_fixture("invalid_yaml.yaml"))
