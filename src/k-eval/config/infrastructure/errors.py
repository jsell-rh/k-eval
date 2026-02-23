"""Error types raised by config infrastructure."""


class MissingEnvVarsError(Exception):
    """Raised when one or more required environment variables are not set."""

    def __init__(self, missing_vars: list[str]) -> None:
        self.missing_vars = missing_vars
        var_list = ", ".join(sorted(missing_vars))
        super().__init__(
            f"Failed to load config: missing environment variables: {var_list}"
        )


class ConfigValidationError(Exception):
    """Raised when the loaded config fails semantic validation."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Failed to validate config: {reason}")
