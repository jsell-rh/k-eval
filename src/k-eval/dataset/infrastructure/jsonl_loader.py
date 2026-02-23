"""JSONL dataset loader — reads a dataset file and returns typed Sample objects."""

import json

from config.domain.dataset import DatasetConfig
from dataset.domain.observer import DatasetObserver
from dataset.domain.sample import Sample
from dataset.infrastructure.errors import DatasetLoadError


class JsonlDatasetLoader:
    """Loads a JSONL dataset file and returns a list of Sample value objects."""

    def __init__(self, observer: DatasetObserver) -> None:
        self._observer = observer

    def load(self, config: DatasetConfig) -> list[Sample]:
        """
        Load all samples from the JSONL file described by config.

        Emits observer events as loading progresses. Collects ALL per-line errors
        before raising a single DatasetLoadError listing every issue found.

        Raises:
            DatasetLoadError: if the file is not found, any line is invalid JSON,
                or any line is missing the configured question or answer key.
        """
        self._observer.dataset_loading_started(
            path=config.path,
            question_key=config.question_key,
            answer_key=config.answer_key,
        )

        try:
            lines = self._read_lines(path=config.path)
        except FileNotFoundError:
            reason = f"file not found: {config.path}"
            self._observer.dataset_loading_failed(path=config.path, reason=reason)
            raise DatasetLoadError(reason=reason)

        samples, errors = self._parse_lines(
            lines=lines,
            question_key=config.question_key,
            answer_key=config.answer_key,
        )

        if errors:
            reason = "; ".join(errors)
            self._observer.dataset_loading_failed(path=config.path, reason=reason)
            raise DatasetLoadError(reason=reason)

        self._observer.dataset_loading_completed(
            path=config.path,
            total_samples=len(samples),
        )
        return samples

    def _read_lines(self, path: str) -> list[str]:
        """Open the file and return all non-empty lines."""
        with open(path, encoding="utf-8") as fh:
            return [line for line in fh if line.strip()]

    def _parse_lines(
        self,
        lines: list[str],
        question_key: str,
        answer_key: str,
    ) -> tuple[list[Sample], list[str]]:
        """Parse each line into a Sample, collecting errors without aborting early."""
        samples: list[Sample] = []
        errors: list[str] = []

        for index, line in enumerate(lines):
            result = self._parse_line(
                line=line,
                index=index,
                question_key=question_key,
                answer_key=answer_key,
            )
            if isinstance(result, str):
                errors.append(result)
            else:
                samples.append(result)
                self._observer.dataset_sample_loaded(sample_id=result.id)

        return samples, errors

    def _parse_line(
        self,
        line: str,
        index: int,
        question_key: str,
        answer_key: str,
    ) -> Sample | str:
        """
        Parse a single JSONL line into a Sample.

        Returns a Sample on success, or an error string describing the problem.
        """
        try:
            data: dict[str, object] = json.loads(line)
        except json.JSONDecodeError as exc:
            return f"line {index}: invalid JSON: {exc}"

        missing: list[str] = []
        if question_key not in data:
            missing.append(question_key)
        if answer_key not in data:
            missing.append(answer_key)

        if missing:
            keys = ", ".join(f"'{k}'" for k in missing)
            return f"line {index}: missing key(s) {keys}"

        return Sample(
            id=str(index),
            question=str(data[question_key]),
            answer=str(data[answer_key]),
        )
