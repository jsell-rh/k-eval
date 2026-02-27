"""Tests verifying the KEvalError type hierarchy."""

from pathlib import Path

from k_eval.agent.infrastructure.errors import (
    McpToolSuccessAbsentError,
    McpToolUseAbsentError,
)
from k_eval.config.infrastructure.errors import (
    ConfigLoadError,
    ConfigValidationError,
    MissingEnvVarsError,
)
from k_eval.core.errors import KEvalError
from k_eval.dataset.infrastructure.errors import DatasetLoadError


class TestKEvalErrorHierarchy:
    """All k-eval-specific exceptions inherit from KEvalError."""

    def test_missing_env_vars_error_is_k_eval_error(self) -> None:
        error = MissingEnvVarsError(missing_vars=["MY_VAR"])
        assert isinstance(error, KEvalError)

    def test_config_validation_error_is_k_eval_error(self) -> None:
        error = ConfigValidationError(reason="bad value")
        assert isinstance(error, KEvalError)

    def test_config_load_error_is_k_eval_error(self) -> None:
        error = ConfigLoadError(path=Path("/some/config.yaml"))
        assert isinstance(error, KEvalError)

    def test_dataset_load_error_is_k_eval_error(self) -> None:
        error = DatasetLoadError(reason="file not found")
        assert isinstance(error, KEvalError)

    def test_k_eval_error_is_exception(self) -> None:
        error = KEvalError("test")
        assert isinstance(error, Exception)


class TestMcpToolUseAbsentError:
    """McpToolUseAbsentError is a retriable KEvalError."""

    def test_is_keval_error(self) -> None:
        error = McpToolUseAbsentError(condition="with-graph", sample_idx=0)
        assert isinstance(error, KEvalError)

    def test_is_retriable(self) -> None:
        error = McpToolUseAbsentError(condition="with-graph", sample_idx=0)
        assert error.retriable is True

    def test_message_starts_with_failed(self) -> None:
        error = McpToolUseAbsentError(condition="baseline", sample_idx=5)
        assert str(error).startswith("Failed to ")

    def test_message_includes_condition_name(self) -> None:
        error = McpToolUseAbsentError(condition="my-condition", sample_idx=3)
        assert "my-condition" in str(error)


class TestMcpToolSuccessAbsentError:
    """McpToolSuccessAbsentError is a retriable KEvalError."""

    def test_is_keval_error(self) -> None:
        error = McpToolSuccessAbsentError(condition="with-graph", sample_idx=0)
        assert isinstance(error, KEvalError)

    def test_is_retriable(self) -> None:
        error = McpToolSuccessAbsentError(condition="with-graph", sample_idx=0)
        assert error.retriable is True

    def test_message_starts_with_failed(self) -> None:
        error = McpToolSuccessAbsentError(condition="baseline", sample_idx=5)
        assert str(error).startswith("Failed to ")

    def test_message_includes_condition_name(self) -> None:
        error = McpToolSuccessAbsentError(condition="my-condition", sample_idx=3)
        assert "my-condition" in str(error)
