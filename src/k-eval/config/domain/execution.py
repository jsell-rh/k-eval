"""Execution configuration models."""

from pydantic import BaseModel


class RetryConfig(BaseModel, frozen=True):
    max_attempts: int
    initial_backoff_seconds: int
    backoff_multiplier: int


class ExecutionConfig(BaseModel, frozen=True):
    num_samples: int
    max_concurrent: int
    retry: RetryConfig
