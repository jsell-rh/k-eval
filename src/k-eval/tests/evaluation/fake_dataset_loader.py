"""FakeDatasetLoader — in-memory DatasetLoader implementation for use in tests."""

from config.domain.dataset import DatasetConfig
from dataset.domain.load_result import DatasetLoadResult
from dataset.domain.sample import Sample


class FakeDatasetLoader:
    """Satisfies the DatasetLoader protocol. Returns canned samples and a fake SHA-256."""

    def __init__(self, samples: list[Sample], sha256: str = "fake-sha256") -> None:
        self._samples = samples
        self._sha256 = sha256

    def load(self, config: DatasetConfig) -> DatasetLoadResult:
        return DatasetLoadResult(samples=self._samples, sha256=self._sha256)
