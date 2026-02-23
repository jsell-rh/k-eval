"""DatasetLoadResult — the result of loading a dataset, including samples and integrity hash."""

from pydantic import BaseModel, Field

from dataset.domain.sample import Sample


class DatasetLoadResult(BaseModel, frozen=True):
    """Immutable value object returned by a DatasetLoader.

    Carries both the parsed samples and the SHA-256 hex digest of the raw file
    bytes, allowing callers to record which exact dataset version was used.
    """

    samples: list[Sample]
    sha256: str = Field(min_length=1)
