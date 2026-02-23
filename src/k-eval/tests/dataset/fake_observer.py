"""Fake DatasetObserver for use in tests — records events without mocking."""


class FakeDatasetObserver:
    def __init__(self) -> None:
        self.loading_started: list[dict[str, str]] = []
        self.samples_loaded: list[dict[str, str]] = []
        self.loading_completed: list[dict[str, str]] = []
        self.loading_failed: list[dict[str, str]] = []

    def dataset_loading_started(
        self, path: str, question_key: str, answer_key: str
    ) -> None:
        self.loading_started.append(
            {"path": path, "question_key": question_key, "answer_key": answer_key}
        )

    def dataset_sample_loaded(self, sample_id: str) -> None:
        self.samples_loaded.append({"sample_id": sample_id})

    def dataset_loading_completed(self, path: str, total_samples: int) -> None:
        self.loading_completed.append(
            {"path": path, "total_samples": str(total_samples)}
        )

    def dataset_loading_failed(self, path: str, reason: str) -> None:
        self.loading_failed.append({"path": path, "reason": reason})
