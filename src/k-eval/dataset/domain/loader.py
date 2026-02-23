"""DatasetLoader Protocol — structural interface for loading dataset samples."""

from typing import Protocol

from config.domain.dataset import DatasetConfig
from dataset.domain.sample import Sample


class DatasetLoader(Protocol):
    """Loads a list of Sample objects from a dataset described by DatasetConfig."""

    def load(self, config: DatasetConfig) -> list[Sample]: ...
