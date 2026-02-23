"""Structlog implementation of the DatasetObserver port."""

import structlog


class StructlogDatasetObserver:
    """Delegates dataset domain events to structlog.

    Satisfies the DatasetObserver protocol structurally.
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger()

    def dataset_loading_started(
        self, path: str, question_key: str, answer_key: str
    ) -> None:
        self._log.info(
            "dataset.loading_started",
            path=path,
            question_key=question_key,
            answer_key=answer_key,
        )

    def dataset_sample_loaded(self, sample_id: str) -> None:
        self._log.debug("dataset.sample_loaded", sample_id=sample_id)

    def dataset_loading_completed(self, path: str, total_samples: int) -> None:
        self._log.info(
            "dataset.loading_completed",
            path=path,
            total_samples=total_samples,
        )

    def dataset_loading_failed(self, path: str, reason: str) -> None:
        self._log.error("dataset.loading_failed", path=path, reason=reason)
