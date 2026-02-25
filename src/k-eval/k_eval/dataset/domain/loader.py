"""DatasetLoader Protocol — structural interface for loading dataset samples."""

from typing import Protocol

from k_eval.config.domain.dataset import DatasetConfig
from k_eval.dataset.domain.load_result import DatasetLoadResult


class DatasetLoader(Protocol):
    """Loads samples from a dataset described by DatasetConfig, returning a DatasetLoadResult."""

    def load(self, config: DatasetConfig) -> DatasetLoadResult: ...
