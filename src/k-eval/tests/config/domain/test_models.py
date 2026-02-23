"""Tests for validation constraints on config domain models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from config.domain.agent import AgentConfig
from config.domain.condition import ConditionConfig
from config.domain.config import EvalConfig
from config.domain.dataset import DatasetConfig
from config.domain.execution import ExecutionConfig, RetryConfig
from config.domain.judge import JudgeConfig
from config.domain.mcp_server import HttpMcpServer, SseMcpServer, StdioMcpServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_retry() -> RetryConfig:
    return RetryConfig(max_attempts=3, initial_backoff_seconds=1, backoff_multiplier=2)


def _valid_execution() -> ExecutionConfig:
    return ExecutionConfig(num_samples=1, max_concurrent=1, retry=_valid_retry())


def _valid_agent() -> AgentConfig:
    return AgentConfig(type="claude_code_sdk", model="claude-sonnet-4-5")


def _valid_judge() -> JudgeConfig:
    return JudgeConfig(model="vertex_ai/claude-opus-4-5", temperature=0.0)


def _valid_dataset() -> DatasetConfig:
    return DatasetConfig(
        path=Path("./questions.jsonl"),
        question_key="question",
        answer_key="answer",
    )


def _valid_stdio_server() -> StdioMcpServer:
    return StdioMcpServer(type="stdio", command="python", args=["-m", "server"])


def _valid_condition() -> ConditionConfig:
    return ConditionConfig(mcp_servers=[], system_prompt="Answer using your knowledge.")


def _valid_eval_config() -> EvalConfig:
    return EvalConfig(
        name="my-eval",
        version="1",
        dataset=_valid_dataset(),
        agent=_valid_agent(),
        judge=_valid_judge(),
        mcp_servers={},
        conditions={"baseline": _valid_condition()},
        execution=_valid_execution(),
    )


# ---------------------------------------------------------------------------
# ExecutionConfig
# ---------------------------------------------------------------------------


class TestExecutionConfigConstraints:
    """ExecutionConfig rejects invalid numeric ranges."""

    def test_num_samples_zero_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionConfig(num_samples=0, max_concurrent=1, retry=_valid_retry())

    def test_num_samples_negative_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionConfig(num_samples=-1, max_concurrent=1, retry=_valid_retry())

    def test_num_samples_one_is_valid(self) -> None:
        cfg = ExecutionConfig(num_samples=1, max_concurrent=1, retry=_valid_retry())
        assert cfg.num_samples == 1

    def test_max_concurrent_zero_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionConfig(num_samples=1, max_concurrent=0, retry=_valid_retry())

    def test_max_concurrent_negative_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionConfig(num_samples=1, max_concurrent=-5, retry=_valid_retry())

    def test_max_concurrent_one_is_valid(self) -> None:
        cfg = ExecutionConfig(num_samples=1, max_concurrent=1, retry=_valid_retry())
        assert cfg.max_concurrent == 1


# ---------------------------------------------------------------------------
# RetryConfig
# ---------------------------------------------------------------------------


class TestRetryConfigConstraints:
    """RetryConfig rejects invalid numeric ranges."""

    def test_max_attempts_zero_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            RetryConfig(max_attempts=0, initial_backoff_seconds=1, backoff_multiplier=2)

    def test_max_attempts_negative_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            RetryConfig(
                max_attempts=-1, initial_backoff_seconds=1, backoff_multiplier=2
            )

    def test_max_attempts_one_is_valid(self) -> None:
        cfg = RetryConfig(
            max_attempts=1, initial_backoff_seconds=1, backoff_multiplier=2
        )
        assert cfg.max_attempts == 1

    def test_initial_backoff_seconds_negative_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            RetryConfig(
                max_attempts=3, initial_backoff_seconds=-1, backoff_multiplier=2
            )

    def test_initial_backoff_seconds_zero_is_valid(self) -> None:
        cfg = RetryConfig(
            max_attempts=3, initial_backoff_seconds=0, backoff_multiplier=2
        )
        assert cfg.initial_backoff_seconds == 0

    def test_backoff_multiplier_zero_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            RetryConfig(max_attempts=3, initial_backoff_seconds=1, backoff_multiplier=0)

    def test_backoff_multiplier_negative_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            RetryConfig(
                max_attempts=3, initial_backoff_seconds=1, backoff_multiplier=-2
            )

    def test_backoff_multiplier_one_is_valid(self) -> None:
        cfg = RetryConfig(
            max_attempts=3, initial_backoff_seconds=1, backoff_multiplier=1
        )
        assert cfg.backoff_multiplier == 1


# ---------------------------------------------------------------------------
# JudgeConfig
# ---------------------------------------------------------------------------


class TestJudgeConfigConstraints:
    """JudgeConfig rejects empty model strings and negative temperatures."""

    def test_empty_model_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            JudgeConfig(model="", temperature=0.0)

    def test_negative_temperature_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            JudgeConfig(model="claude-opus", temperature=-0.1)

    def test_zero_temperature_is_valid(self) -> None:
        cfg = JudgeConfig(model="claude-opus", temperature=0.0)
        assert cfg.temperature == 0.0

    def test_positive_temperature_is_valid(self) -> None:
        cfg = JudgeConfig(model="claude-opus", temperature=1.0)
        assert cfg.temperature == 1.0

    def test_non_empty_model_is_valid(self) -> None:
        cfg = JudgeConfig(model="claude-opus", temperature=0.0)
        assert cfg.model == "claude-opus"


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfigConstraints:
    """AgentConfig rejects empty type and model strings."""

    def test_empty_type_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(type="", model="claude-sonnet-4-5")

    def test_empty_model_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(type="claude_code_sdk", model="")

    def test_valid_type_and_model_accepted(self) -> None:
        cfg = AgentConfig(type="claude_code_sdk", model="claude-sonnet-4-5")
        assert cfg.type == "claude_code_sdk"
        assert cfg.model == "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------


class TestDatasetConfigConstraints:
    """DatasetConfig rejects empty question_key and answer_key."""

    def test_empty_question_key_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            DatasetConfig(
                path=Path("./data.jsonl"),
                question_key="",
                answer_key="answer",
            )

    def test_empty_answer_key_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            DatasetConfig(
                path=Path("./data.jsonl"),
                question_key="question",
                answer_key="",
            )

    def test_valid_keys_accepted(self) -> None:
        cfg = DatasetConfig(
            path=Path("./data.jsonl"),
            question_key="question",
            answer_key="answer",
        )
        assert cfg.question_key == "question"
        assert cfg.answer_key == "answer"


# ---------------------------------------------------------------------------
# EvalConfig
# ---------------------------------------------------------------------------


class TestEvalConfigConstraints:
    """EvalConfig rejects empty name/version and empty conditions dict."""

    def test_empty_name_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            EvalConfig(
                name="",
                version="1",
                dataset=_valid_dataset(),
                agent=_valid_agent(),
                judge=_valid_judge(),
                mcp_servers={},
                conditions={"baseline": _valid_condition()},
                execution=_valid_execution(),
            )

    def test_empty_version_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            EvalConfig(
                name="my-eval",
                version="",
                dataset=_valid_dataset(),
                agent=_valid_agent(),
                judge=_valid_judge(),
                mcp_servers={},
                conditions={"baseline": _valid_condition()},
                execution=_valid_execution(),
            )

    def test_empty_conditions_dict_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            EvalConfig(
                name="my-eval",
                version="1",
                dataset=_valid_dataset(),
                agent=_valid_agent(),
                judge=_valid_judge(),
                mcp_servers={},
                conditions={},
                execution=_valid_execution(),
            )

    def test_valid_config_accepted(self) -> None:
        cfg = _valid_eval_config()
        assert cfg.name == "my-eval"
        assert cfg.version == "1"
        assert len(cfg.conditions) == 1


# ---------------------------------------------------------------------------
# ConditionConfig
# ---------------------------------------------------------------------------


class TestConditionConfigConstraints:
    """ConditionConfig rejects empty system_prompt."""

    def test_empty_system_prompt_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ConditionConfig(mcp_servers=[], system_prompt="")

    def test_non_empty_system_prompt_accepted(self) -> None:
        cfg = ConditionConfig(
            mcp_servers=[], system_prompt="Answer using your knowledge."
        )
        assert cfg.system_prompt == "Answer using your knowledge."


# ---------------------------------------------------------------------------
# StdioMcpServer
# ---------------------------------------------------------------------------


class TestStdioMcpServerConstraints:
    """StdioMcpServer rejects empty command strings."""

    def test_empty_command_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            StdioMcpServer(type="stdio", command="")

    def test_non_empty_command_accepted(self) -> None:
        server = StdioMcpServer(type="stdio", command="python")
        assert server.command == "python"

    def test_command_with_args_accepted(self) -> None:
        server = StdioMcpServer(
            type="stdio", command="python", args=["-m", "kartograph.mcp"]
        )
        assert server.command == "python"
        assert server.args == ["-m", "kartograph.mcp"]


# ---------------------------------------------------------------------------
# SseMcpServer
# ---------------------------------------------------------------------------


class TestSseMcpServerConstraints:
    """SseMcpServer rejects empty url strings."""

    def test_empty_url_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            SseMcpServer(type="sse", url="")

    def test_non_empty_url_accepted(self) -> None:
        server = SseMcpServer(type="sse", url="http://localhost:8080/sse")
        assert server.url == "http://localhost:8080/sse"


# ---------------------------------------------------------------------------
# HttpMcpServer
# ---------------------------------------------------------------------------


class TestHttpMcpServerConstraints:
    """HttpMcpServer rejects empty url strings."""

    def test_empty_url_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            HttpMcpServer(type="http", url="")

    def test_non_empty_url_accepted(self) -> None:
        server = HttpMcpServer(type="http", url="http://localhost:9000/mcp")
        assert server.url == "http://localhost:9000/mcp"
