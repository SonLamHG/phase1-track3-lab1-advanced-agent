from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from .runtime import get_runtime
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord
from .utils import normalize_answer


@lru_cache(maxsize=1)
def _runtime():
    # Lazy so callers can set LLM_PROVIDER (e.g. via load_dotenv()) before the first run.
    return get_runtime()


def _failure_mode_by_qid() -> dict[str, str]:
    return getattr(_runtime(), "FAILURE_MODE_BY_QID", {})


def classify_failure(reason: str) -> str:
    """Heuristic mapping of evaluator critique to a schema failure_mode label.
    Patterns observed empirically in gpt-5-nano evaluator output on HotpotQA."""
    r = (reason or "").lower()

    # Reflexion looping cues (rare for max_attempts=3, but possible)
    if any(kw in r for kw in ("loop", "repeating", "stuck", "circular")):
        return "looping"

    # Incomplete: omits a required component / first-hop only / partial answer
    if any(kw in r for kw in (
        "omit", "omits", "missing", "incomplete", "partial",
        "first hop", "first-hop", "stopped at", "did not complete",
        "only the first", "lacks", "does not include",
    )):
        return "incomplete_multi_hop"

    # Entity drift: named the wrong thing in the same conceptual slot
    if any(kw in r for kw in (
        "names a", "names the", "wrong entity", "different entity",
        "instead of", "rather than", "not the ",
        "drift", "incorrect entity", "wrong second",
    )):
        return "entity_drift"

    return "wrong_final_answer"


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        seen_norm_answers: list[str] = []  # for adaptive_max_attempts looping detection
        final_answer = ""
        final_score = 0
        adaptive_loop_stop = False

        rt = _runtime()
        for attempt_id in range(1, self.max_attempts + 1):
            answer, actor_stats = rt.actor_answer(
                example, attempt_id, self.agent_type, reflection_memory
            )
            judge, eval_stats = rt.evaluator(example, answer)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=actor_stats.total_tokens + eval_stats.total_tokens,
                latency_ms=actor_stats.latency_ms + eval_stats.latency_ms,
            )
            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                break

            # adaptive_max_attempts: if Reflexion produces the same wrong answer twice,
            # further reflection isn't producing novel hypotheses — stop early.
            norm = normalize_answer(answer)
            if self.agent_type == "reflexion" and norm in seen_norm_answers:
                traces.append(trace)
                adaptive_loop_stop = True
                break
            seen_norm_answers.append(norm)

            # Reflexion: failed attempt + more attempts remain -> reflect and push lesson.
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                entry, reflect_stats = rt.reflector(example, attempt_id, judge)
                trace.token_estimate += reflect_stats.total_tokens
                trace.latency_ms += reflect_stats.latency_ms
                trace.reflection = entry
                reflections.append(entry)
                reflection_memory.append(
                    f"Attempt {entry.attempt_id} failed: {entry.failure_reason} "
                    f"Lesson: {entry.lesson} Next strategy: {entry.next_strategy}"
                )
            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        if final_score == 1:
            failure_mode = "none"
        elif adaptive_loop_stop:
            failure_mode = "looping"
        else:
            # Prefer per-qid override (mock fixtures); else classify from evaluator's reason.
            override = _failure_mode_by_qid().get(example.qid)
            last_reason = traces[-1].reason if traces else ""
            failure_mode = override or classify_failure(last_reason)
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
