"""Structlog implementation of the ConfigObserver port."""

import structlog


class StructlogConfigObserver:
    """Delegates config domain events to structlog.

    Satisfies the ConfigObserver protocol structurally.
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger()

    def config_loaded(self, name: str, version: str) -> None:
        self._log.info("config.loaded", name=name, version=version)

    def config_judge_temperature_warning(self, temperature: float) -> None:
        self._log.warning(
            "config.judge_temperature_warning",
            temperature=temperature,
            message="Judge temperature > 0.0 may produce non-deterministic scoring",
        )
