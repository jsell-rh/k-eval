"""Execution configuration models."""

from pydantic import BaseModel, Field


class RetryConfig(BaseModel, frozen=True):
    max_attempts: int = Field(ge=1)
    initial_backoff_seconds: int = Field(ge=0)
    backoff_multiplier: int = Field(ge=1)


class ExecutionConfig(BaseModel, frozen=True):
    num_repetitions: int = Field(ge=1)
    max_concurrent: int = Field(ge=1)
    retry: RetryConfig
