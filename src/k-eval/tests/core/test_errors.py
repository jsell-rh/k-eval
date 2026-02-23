"""Tests verifying the KEvalError type hierarchy."""

from pathlib import Path

from config.infrastructure.errors import (
    ConfigLoadError,
    ConfigValidationError,
    MissingEnvVarsError,
)
from core.errors import KEvalError
from dataset.infrastructure.errors import DatasetLoadError


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
