"""Observer port for the dataset domain — defines events in domain language."""

from typing import Protocol


class DatasetObserver(Protocol):
    def dataset_loading_started(
        self, path: str, question_key: str, answer_key: str
    ) -> None: ...

    def dataset_sample_loaded(self, sample_id: str) -> None: ...

    def dataset_loading_completed(self, path: str, total_samples: int) -> None: ...

    def dataset_loading_failed(self, path: str, reason: str) -> None: ...
