"""Error types raised by config infrastructure."""

from pathlib import Path

from core.errors import KEvalError


class MissingEnvVarsError(KEvalError):
    """Raised when one or more required environment variables are not set."""

    def __init__(self, missing_vars: list[str]) -> None:
        self.missing_vars = missing_vars
        var_list = ", ".join(sorted(missing_vars))
        super().__init__(
            f"Failed to load config: missing environment variables: {var_list}"
        )


class ConfigValidationError(KEvalError):
    """Raised when the loaded config fails semantic validation."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Failed to validate config: {reason}")


class ConfigLoadError(KEvalError):
    """Raised when the config file cannot be opened or read."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"Failed to load config: file not found: {path}")
