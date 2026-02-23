"""YAML config loader — parses, interpolates env vars, validates, and emits observer events."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from config.domain.config import EvalConfig
from config.domain.observer import ConfigObserver
from config.infrastructure.env_interpolation import collect_missing_vars, interpolate
from config.infrastructure.errors import ConfigValidationError, MissingEnvVarsError


class YamlConfigLoader:
    """Loads, interpolates, validates, and returns an EvalConfig from a YAML file."""

    def __init__(self, observer: ConfigObserver) -> None:
        self._observer = observer

    def load(self, path: Path) -> EvalConfig:
        """
        Load, interpolate, validate, and return an EvalConfig from a YAML file.

        Raises:
            MissingEnvVarsError: if any ${ENV_VAR} references are unset (all collected first).
            ConfigValidationError: if any condition references an unknown MCP server name.
            yaml.YAMLError: if the file is not valid YAML.
            pydantic.ValidationError: if the schema is violated.
        """
        raw = _parse_yaml(path=path)
        _check_missing_env_vars(raw=raw)
        interpolated = _interpolate(raw=raw)
        resolved = _resolve_condition_server_refs(interpolated=interpolated)
        cfg = _build_config(resolved=resolved)
        _emit_warnings(cfg=cfg, observer=self._observer)
        self._observer.config_loaded(name=cfg.name, version=cfg.version)
        return cfg


def _parse_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _check_missing_env_vars(raw: Any) -> None:
    """Raise MissingEnvVarsError if any ${ENV_VAR} references in raw are unset."""
    missing = collect_missing_vars(raw)
    if missing:
        raise MissingEnvVarsError(missing)


def _interpolate(raw: Any) -> Any:
    """Return a fully interpolated copy of raw with all ${ENV_VAR} substituted."""
    return interpolate(raw)


def _resolve_condition_server_refs(interpolated: Any) -> Any:
    """
    Validate condition MCP server references and replace each name with a resolved dict.

    Replaces each condition's ``mcp_servers: [str, ...]`` with
    ``mcp_servers: [{"name": str, "config": dict}, ...]`` so that Pydantic can
    validate the full ``EvalConfig`` — including ``ConditionMcpServer`` — in one pass.

    Raises:
        ConfigValidationError: listing ALL invalid references across ALL conditions
            before raising (not just the first one).
    """
    mcp_servers_raw: dict[str, Any] = interpolated.get("mcp_servers", {}) or {}
    conditions_raw: dict[str, Any] = interpolated.get("conditions", {}) or {}
    defined_names = set(mcp_servers_raw.keys())

    unknown: list[str] = []
    for condition_name, condition_data in conditions_raw.items():
        server_names: list[str] = condition_data.get("mcp_servers", []) or []
        for server_name in server_names:
            if server_name not in defined_names:
                unknown.append(
                    f"condition '{condition_name}' references unknown MCP server"
                    f" '{server_name}'"
                )

    if unknown:
        detail = "; ".join(unknown)
        raise ConfigValidationError(detail)

    # All references are valid — replace names with resolved dicts for Pydantic.
    resolved_conditions: dict[str, Any] = {}
    for condition_name, condition_data in conditions_raw.items():
        server_names = condition_data.get("mcp_servers", []) or []
        resolved_servers = [
            {"name": name, "config": mcp_servers_raw[name]} for name in server_names
        ]
        resolved_conditions[condition_name] = {
            **condition_data,
            "mcp_servers": resolved_servers,
        }

    return {**interpolated, "conditions": resolved_conditions}


def _build_config(resolved: Any) -> EvalConfig:
    try:
        return EvalConfig.model_validate(resolved)
    except ValidationError as exc:
        raise ConfigValidationError(str(exc)) from exc


def _emit_warnings(cfg: EvalConfig, observer: ConfigObserver) -> None:
    if cfg.judge.temperature > 0.0:
        observer.config_judge_temperature_warning(cfg.judge.temperature)
