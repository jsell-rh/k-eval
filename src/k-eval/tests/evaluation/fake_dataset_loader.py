"""FakeDatasetLoader — in-memory DatasetLoader implementation for use in tests."""

from config.domain.dataset import DatasetConfig
from dataset.domain.sample import Sample


class FakeDatasetLoader:
    """Satisfies the DatasetLoader protocol. Returns a canned list of samples."""

    def __init__(self, samples: list[Sample]) -> None:
        self._samples = samples

    def load(self, config: DatasetConfig) -> list[Sample]:
        return self._samples
