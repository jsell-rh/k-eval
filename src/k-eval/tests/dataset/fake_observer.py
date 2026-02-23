"""Fake DatasetObserver for use in tests — records events without mocking."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadingStartedEvent:
    path: str
    question_key: str
    answer_key: str


@dataclass(frozen=True)
class SampleLoadedEvent:
    sample_id: str


@dataclass(frozen=True)
class LoadingCompletedEvent:
    path: str
    total_samples: int


@dataclass(frozen=True)
class LoadingFailedEvent:
    path: str
    reason: str


class FakeDatasetObserver:
    def __init__(self) -> None:
        self.loading_started: list[LoadingStartedEvent] = []
        self.samples_loaded: list[SampleLoadedEvent] = []
        self.loading_completed: list[LoadingCompletedEvent] = []
        self.loading_failed: list[LoadingFailedEvent] = []

    def dataset_loading_started(
        self, path: str, question_key: str, answer_key: str
    ) -> None:
        self.loading_started.append(
            LoadingStartedEvent(
                path=path, question_key=question_key, answer_key=answer_key
            )
        )

    def dataset_sample_loaded(self, sample_id: str) -> None:
        self.samples_loaded.append(SampleLoadedEvent(sample_id=sample_id))

    def dataset_loading_completed(self, path: str, total_samples: int) -> None:
        self.loading_completed.append(
            LoadingCompletedEvent(path=path, total_samples=total_samples)
        )

    def dataset_loading_failed(self, path: str, reason: str) -> None:
        self.loading_failed.append(LoadingFailedEvent(path=path, reason=reason))
