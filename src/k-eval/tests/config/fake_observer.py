"""Fake ConfigObserver for use in tests — records events without mocking."""


class FakeConfigObserver:
    def __init__(self) -> None:
        self.loaded: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []

    def config_loaded(self, name: str, version: str) -> None:
        self.loaded.append({"name": name, "version": version})

    def config_judge_temperature_warning(self, temperature: float) -> None:
        self.warnings.append({"temperature": str(temperature)})
