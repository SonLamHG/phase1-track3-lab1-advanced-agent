"""OpenAI-backed actor / evaluator / reflector using Structured Outputs.

Required env vars (.env is auto-loaded):
    OPENAI_API_KEY       your OpenAI API key

Optional env vars:
    LLM_ACTOR_MODEL      default "gpt-4o-mini"
    LLM_EVAL_MODEL       default "gpt-4o-mini"
    LLM_REFLECT_MODEL    default "gpt-4o-mini"
    LLM_TEMPERATURE      default "0.0" (actor/reflector). Evaluator is always 0.0.
    LLM_MAX_RETRIES      default "3"
"""
from __future__ import annotations
import os
import time
from functools import lru_cache, wraps

from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .runtime import LLMStats
from .schemas import JudgeResult, QAExample, ReflectionEntry

load_dotenv()

ACTOR_MODEL = os.environ.get("LLM_ACTOR_MODEL", "gpt-4o-mini")
EVAL_MODEL = os.environ.get("LLM_EVAL_MODEL", "gpt-4o-mini")
REFLECT_MODEL = os.environ.get("LLM_REFLECT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
# Only used for reasoning models (gpt-5-*, o1/o3/o4): minimal | low | medium | high
REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "low")


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _extra_params(model: str, temperature: float) -> dict:
    # Reasoning models reject `temperature` and instead take `reasoning_effort`.
    if _is_reasoning_model(model):
        return {"reasoning_effort": REASONING_EFFORT}
    return {"temperature": temperature}

# With a real LLM we don't know a priori which qids will fail, so this map stays
# empty and agents.py falls back to "wrong_final_answer" when score == 0.
FAILURE_MODE_BY_QID: dict[str, str] = {}


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key, "
            "or export OPENAI_API_KEY=sk-... before running."
        )
    return OpenAI()


def _retry(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        delay = 1.0
        for attempt in range(MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except (RateLimitError, APITimeoutError, APIError):
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(delay)
                delay *= 2
    return wrapped


def _format_context(example: QAExample) -> str:
    return "\n\n".join(
        f"[{i}] Title: {c.title}\nText: {c.text}"
        for i, c in enumerate(example.context, start=1)
    )


def _stats_from(resp, latency_ms: int) -> LLMStats:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return LLMStats(latency_ms=latency_ms)
    return LLMStats(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
    )


@_retry
def actor_answer(
    example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]
) -> tuple[str, LLMStats]:
    memory_block = ""
    if reflection_memory:
        lines = "\n".join(f"- {m}" for m in reflection_memory)
        memory_block = f"\n\nREFLECTION_MEMORY (lessons from previous failed attempts on this same question):\n{lines}"

    user = (
        f"QUESTION: {example.question}\n\n"
        f"CONTEXT:\n{_format_context(example)}"
        f"{memory_block}\n\n"
        f"Return ONLY the final answer as a short phrase. No explanation."
    )

    t0 = time.perf_counter()
    resp = _client().chat.completions.create(
        model=ACTOR_MODEL,
        messages=[
            {"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user", "content": user},
        ],
        **_extra_params(ACTOR_MODEL, TEMPERATURE),
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    answer = (resp.choices[0].message.content or "").strip()
    return answer, _stats_from(resp, latency_ms)


@_retry
def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, LLMStats]:
    user = (
        f"QUESTION: {example.question}\n"
        f"GOLD_ANSWER: {example.gold_answer}\n"
        f"PREDICTED_ANSWER: {answer}\n\n"
        f"Return the JSON object as specified."
    )

    t0 = time.perf_counter()
    resp = _client().beta.chat.completions.parse(
        model=EVAL_MODEL,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format=JudgeResult,
        **_extra_params(EVAL_MODEL, 0.0),
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    judge = resp.choices[0].message.parsed
    if judge is None:
        judge = JudgeResult(
            score=0,
            reason="parse_failed: evaluator returned invalid structured output",
            missing_evidence=[],
            spurious_claims=[],
        )
    return judge, _stats_from(resp, latency_ms)


@_retry
def reflector(
    example: QAExample, attempt_id: int, judge: JudgeResult
) -> tuple[ReflectionEntry, LLMStats]:
    prior_strategies = []  # agents.py currently does not surface prior strategies per attempt;
    # this is a hook for a future extension. Leaving blank keeps prompt correct.

    user = (
        f"QUESTION: {example.question}\n\n"
        f"CONTEXT:\n{_format_context(example)}\n\n"
        f"FAILED_PREDICTION_CRITIQUE:\n"
        f"- score: {judge.score}\n"
        f"- reason: {judge.reason}\n"
        f"- missing_evidence: {judge.missing_evidence}\n"
        f"- spurious_claims: {judge.spurious_claims}\n\n"
        f"CURRENT_ATTEMPT_ID: {attempt_id}\n"
        f"PRIOR_STRATEGIES: {prior_strategies}\n\n"
        f"Return the JSON reflection object as specified."
    )

    t0 = time.perf_counter()
    resp = _client().beta.chat.completions.parse(
        model=REFLECT_MODEL,
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format=ReflectionEntry,
        **_extra_params(REFLECT_MODEL, TEMPERATURE),
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    entry = resp.choices[0].message.parsed
    if entry is None:
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason="parse_failed: reflector returned invalid structured output",
            lesson="Reflector output was unparseable; default to a conservative re-read.",
            next_strategy="Re-read every paragraph carefully and complete each reasoning hop explicitly before answering.",
        )
    return entry, _stats_from(resp, latency_ms)
