"""Runtime selection and shared LLM stats type.

Set env var LLM_PROVIDER to pick the backend:
    LLM_PROVIDER=mock    -> mock_runtime (default, for scaffold/autograde smoke tests)
    LLM_PROVIDER=openai  -> llm_runtime (real OpenAI gpt-4o-mini calls)
"""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class LLMStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def get_runtime():
    """Return a module exposing actor_answer / evaluator / reflector / FAILURE_MODE_BY_QID."""
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()
    if provider == "mock":
        from . import mock_runtime
        return mock_runtime
    if provider == "openai":
        from . import llm_runtime
        return llm_runtime
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Expected one of: mock, openai."
    )
