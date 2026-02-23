"""Tests for JSONL dataset loading infrastructure."""

from pathlib import Path

import pytest

from config.domain.dataset import DatasetConfig
from dataset.infrastructure.errors import DatasetLoadError
from dataset.infrastructure.jsonl_loader import JsonlDatasetLoader
from tests.dataset.fake_observer import FakeDatasetObserver

# Fixtures directory — absolute so tests are location-independent
# __file__ is tests/dataset/infrastructure/test_jsonl_loader.py
# parent.parent.parent is the tests/ directory
FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


def _fixture(name: str) -> Path:
    return FIXTURES / name


def _simple_config() -> DatasetConfig:
    return DatasetConfig(
        path=_fixture("simple_dataset.jsonl"),
        question_key="question",
        answer_key="answer",
    )


def _custom_keys_config() -> DatasetConfig:
    return DatasetConfig(
        path=_fixture("custom_keys_dataset.jsonl"),
        question_key="prompt",
        answer_key="response",
    )


class TestValidDatasetLoading:
    """A valid JSONL dataset loads correctly with all samples populated."""

    def test_loads_correct_number_of_samples(self) -> None:
        observer = FakeDatasetObserver()
        result = JsonlDatasetLoader(observer=observer).load(config=_simple_config())

        assert len(result.samples) == 3

    def test_loads_question_and_answer_fields(self) -> None:
        observer = FakeDatasetObserver()
        result = JsonlDatasetLoader(observer=observer).load(config=_simple_config())

        assert result.samples[0].question == "What is the capital of France?"
        assert result.samples[0].answer == "Paris"

    def test_question_key_and_answer_key_from_config_are_used(self) -> None:
        observer = FakeDatasetObserver()
        result = JsonlDatasetLoader(observer=observer).load(
            config=_custom_keys_config()
        )

        assert result.samples[0].question == "What is the speed of light?"
        assert result.samples[0].answer == "approximately 299,792,458 metres per second"

    def test_sample_id_is_zero_based_line_index_as_string(self) -> None:
        observer = FakeDatasetObserver()
        result = JsonlDatasetLoader(observer=observer).load(config=_simple_config())

        assert result.samples[0].sample_idx == "0"
        assert result.samples[1].sample_idx == "1"
        assert result.samples[2].sample_idx == "2"

    def test_all_samples_loaded_in_order(self) -> None:
        observer = FakeDatasetObserver()
        result = JsonlDatasetLoader(observer=observer).load(config=_simple_config())

        assert result.samples[0].question == "What is the capital of France?"
        assert result.samples[1].question == "What is 2 + 2?"
        assert result.samples[2].question == "Who wrote Hamlet?"

    def test_sha256_is_a_64_character_hex_string(self) -> None:
        observer = FakeDatasetObserver()
        result = JsonlDatasetLoader(observer=observer).load(config=_simple_config())

        assert len(result.sha256) == 64
        assert all(c in "0123456789abcdef" for c in result.sha256)

    def test_sha256_is_stable_across_loads(self) -> None:
        observer = FakeDatasetObserver()
        loader = JsonlDatasetLoader(observer=observer)
        result_a = loader.load(config=_simple_config())
        result_b = loader.load(config=_simple_config())

        assert result_a.sha256 == result_b.sha256

    def test_different_files_have_different_sha256(self) -> None:
        observer = FakeDatasetObserver()
        loader = JsonlDatasetLoader(observer=observer)
        result_simple = loader.load(config=_simple_config())
        result_custom = loader.load(config=_custom_keys_config())

        assert result_simple.sha256 != result_custom.sha256


class TestObserverEvents:
    """Observer events are emitted at the correct points with correct data."""

    def test_dataset_loading_started_emitted_with_correct_path_and_keys(self) -> None:
        observer = FakeDatasetObserver()
        config = _simple_config()
        JsonlDatasetLoader(observer=observer).load(config=config)

        assert len(observer.loading_started) == 1
        assert observer.loading_started[0].path == str(config.path)
        assert observer.loading_started[0].question_key == "question"
        assert observer.loading_started[0].answer_key == "answer"

    def test_dataset_sample_loaded_emitted_once_per_sample(self) -> None:
        observer = FakeDatasetObserver()
        JsonlDatasetLoader(observer=observer).load(config=_simple_config())

        assert len(observer.samples_loaded) == 3

    def test_dataset_sample_loaded_emitted_with_correct_ids(self) -> None:
        observer = FakeDatasetObserver()
        JsonlDatasetLoader(observer=observer).load(config=_simple_config())

        ids = [e.sample_idx for e in observer.samples_loaded]
        assert ids == ["0", "1", "2"]

    def test_dataset_loading_completed_emitted_with_correct_path(self) -> None:
        observer = FakeDatasetObserver()
        config = _simple_config()
        JsonlDatasetLoader(observer=observer).load(config=config)

        assert len(observer.loading_completed) == 1
        assert observer.loading_completed[0].path == str(config.path)

    def test_dataset_loading_completed_emitted_with_correct_total_count(self) -> None:
        observer = FakeDatasetObserver()
        JsonlDatasetLoader(observer=observer).load(config=_simple_config())

        assert observer.loading_completed[0].total_samples == 3


class TestCustomKeys:
    """Custom question_key and answer_key values are respected."""

    def test_custom_keys_load_all_samples(self) -> None:
        observer = FakeDatasetObserver()
        result = JsonlDatasetLoader(observer=observer).load(
            config=_custom_keys_config()
        )

        assert len(result.samples) == 3

    def test_custom_keys_second_sample(self) -> None:
        observer = FakeDatasetObserver()
        result = JsonlDatasetLoader(observer=observer).load(
            config=_custom_keys_config()
        )

        assert result.samples[1].question == "What is the boiling point of water?"
        assert result.samples[1].answer == "100 degrees Celsius at sea level"

    def test_custom_keys_ids_are_zero_based_strings(self) -> None:
        observer = FakeDatasetObserver()
        result = JsonlDatasetLoader(observer=observer).load(
            config=_custom_keys_config()
        )

        assert [s.sample_idx for s in result.samples] == ["0", "1", "2"]


class TestMissingFile:
    """Loading a JSONL file that does not exist raises DatasetLoadError."""

    def test_missing_file_raises_dataset_load_error(self) -> None:
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=Path("/nonexistent/path/dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError):
            JsonlDatasetLoader(observer=observer).load(config=config)

    def test_missing_file_error_message_starts_with_failed(self) -> None:
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=Path("/nonexistent/path/dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError) as exc_info:
            JsonlDatasetLoader(observer=observer).load(config=config)

        assert str(exc_info.value).startswith("Failed to ")

    def test_missing_file_emits_dataset_loading_failed(self) -> None:
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=Path("/nonexistent/path/dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError):
            JsonlDatasetLoader(observer=observer).load(config=config)

        assert len(observer.loading_failed) == 1
        assert observer.loading_failed[0].path == str(config.path)


class TestMissingKeys:
    """Lines missing the question or answer key raise DatasetLoadError listing all bad lines."""

    def test_missing_answer_key_raises_dataset_load_error(self) -> None:
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=_fixture("missing_key_dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError):
            JsonlDatasetLoader(observer=observer).load(config=config)

    def test_missing_keys_error_message_starts_with_failed(self) -> None:
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=_fixture("missing_key_dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError) as exc_info:
            JsonlDatasetLoader(observer=observer).load(config=config)

        assert str(exc_info.value).startswith("Failed to ")

    def test_missing_answer_key_reports_line_number(self) -> None:
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=_fixture("missing_key_dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError) as exc_info:
            JsonlDatasetLoader(observer=observer).load(config=config)

        # line index 1 is missing answer key
        assert "1" in str(exc_info.value)

    def test_missing_question_key_reports_line_number(self) -> None:
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=_fixture("missing_key_dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError) as exc_info:
            JsonlDatasetLoader(observer=observer).load(config=config)

        # line index 2 is missing question key
        assert "2" in str(exc_info.value)

    def test_all_bad_lines_collected_before_raising(self) -> None:
        """All lines with missing keys are reported in a single error."""
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=_fixture("missing_key_dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError) as exc_info:
            JsonlDatasetLoader(observer=observer).load(config=config)

        error_msg = str(exc_info.value)
        # Both bad lines must appear in the single error
        assert "1" in error_msg
        assert "2" in error_msg

    def test_missing_keys_emits_dataset_loading_failed(self) -> None:
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=_fixture("missing_key_dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError):
            JsonlDatasetLoader(observer=observer).load(config=config)

        assert len(observer.loading_failed) == 1


class TestInvalidJson:
    """Lines with invalid JSON raise DatasetLoadError."""

    def test_invalid_json_raises_dataset_load_error(self) -> None:
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=_fixture("invalid_json_dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError):
            JsonlDatasetLoader(observer=observer).load(config=config)

    def test_invalid_json_error_message_starts_with_failed(self) -> None:
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=_fixture("invalid_json_dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError) as exc_info:
            JsonlDatasetLoader(observer=observer).load(config=config)

        assert str(exc_info.value).startswith("Failed to ")

    def test_invalid_json_emits_dataset_loading_failed(self) -> None:
        observer = FakeDatasetObserver()
        config = DatasetConfig(
            path=_fixture("invalid_json_dataset.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        with pytest.raises(DatasetLoadError):
            JsonlDatasetLoader(observer=observer).load(config=config)

        assert len(observer.loading_failed) == 1
