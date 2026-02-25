"""Recursive ${ENV_VAR} interpolation for raw config data."""

import os
import re

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

type RawValue = (
    str | int | float | bool | None | list["RawValue"] | dict[str, "RawValue"]
)


def collect_missing_vars(data: RawValue) -> list[str]:
    """
    Walk the data tree and return the names of all referenced env vars that
    are not currently set.  Every missing var is collected before returning.
    """
    missing: list[str] = []
    _collect(data, missing)
    return missing


def _collect(data: RawValue, missing: list[str]) -> None:
    if isinstance(data, str):
        for match in _ENV_VAR_PATTERN.finditer(data):
            var_name = match.group(1)
            if var_name not in os.environ and var_name not in missing:
                missing.append(var_name)
    elif isinstance(data, list):
        for item in data:
            _collect(item, missing)
    elif isinstance(data, dict):
        for value in data.values():
            _collect(value, missing)


def interpolate(data: RawValue) -> RawValue:
    """
    Recursively substitute all ${ENV_VAR} occurrences with their runtime values.

    Assumes all referenced variables are present in the environment — call
    `collect_missing_vars` first and raise `MissingEnvVarsError` if any are absent.
    """
    if isinstance(data, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ[m.group(1)], data)
    if isinstance(data, list):
        return [interpolate(item) for item in data]
    if isinstance(data, dict):
        return {key: interpolate(value) for key, value in data.items()}
    return data
