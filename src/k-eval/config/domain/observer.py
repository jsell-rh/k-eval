"""Observer port for the config domain — defines events in domain language."""

from typing import Protocol


class ConfigObserver(Protocol):
    def config_loaded(self, name: str, version: str) -> None: ...

    def config_judge_temperature_warning(self, temperature: float) -> None: ...
