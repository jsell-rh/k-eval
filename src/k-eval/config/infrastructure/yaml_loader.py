"""YAML config loader — parses, interpolates env vars, validates, and emits observer events."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from config.domain.config import EvalConfig
from config.domain.observer import ConfigObserver
from config.infrastructure.env_interpolation import collect_missing_vars, interpolate
from config.infrastructure.errors import ConfigValidationError, MissingEnvVarsError


def load_config(path: Path, observer: ConfigObserver) -> EvalConfig:
    """
    Load, interpolate, validate, and return an EvalConfig from a YAML file.

    Raises:
        MissingEnvVarsError: if any ${ENV_VAR} references are unset (all collected first).
        ConfigValidationError: if condition references an unknown MCP server name.
        yaml.YAMLError: if the file is not valid YAML.
        pydantic.ValidationError: if the schema is violated.
    """
    raw = _parse_yaml(path)
    _interpolate_env_vars(raw)
    cfg = _build_config(raw)
    _validate_condition_server_refs(cfg)
    _emit_warnings(cfg, observer)
    observer.config_loaded(name=cfg.name, version=cfg.version)
    return cfg


def _parse_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _interpolate_env_vars(raw: Any) -> Any:
    """Mutates nothing — returns the interpolated tree.  Raises if vars are missing."""
    missing = collect_missing_vars(raw)
    if missing:
        raise MissingEnvVarsError(missing)
    return interpolate(raw)


def _build_config(raw: Any) -> EvalConfig:
    # _interpolate_env_vars returns the interpolated copy but we need to pass it to
    # Pydantic.  Re-run interpolation (env vars are present now) to get the final dict.
    interpolated = interpolate(raw)
    try:
        return EvalConfig.model_validate(interpolated)
    except ValidationError as exc:
        raise ConfigValidationError(str(exc)) from exc


def _validate_condition_server_refs(cfg: EvalConfig) -> None:
    """Ensure every server name referenced by a condition exists in mcp_servers."""
    defined_servers = set(cfg.mcp_servers.keys())
    unknown: list[str] = []

    for condition_name, condition in cfg.conditions.items():
        for server_name in condition.mcp_servers:
            if server_name not in defined_servers:
                unknown.append(
                    f"condition '{condition_name}' references unknown MCP server '{server_name}'"
                )

    if unknown:
        detail = "; ".join(unknown)
        raise ConfigValidationError(detail)


def _emit_warnings(cfg: EvalConfig, observer: ConfigObserver) -> None:
    if cfg.judge.temperature > 0.0:
        observer.config_judge_temperature_warning(cfg.judge.temperature)
