from __future__ import annotations
import time
from .runtime import LLMStats
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}


def _estimate_tokens(*texts: str) -> int:
    # ~4 chars/token heuristic — matches what a real tokenizer would report within ~30%.
    return max(1, sum(len(t) for t in texts if t) // 4)


def actor_answer(
    example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]
) -> tuple[str, LLMStats]:
    t0 = time.perf_counter()
    if example.qid not in FIRST_ATTEMPT_WRONG:
        answer = example.gold_answer
    elif agent_type == "react":
        answer = FIRST_ATTEMPT_WRONG[example.qid]
    elif attempt_id == 1 and not reflection_memory:
        answer = FIRST_ATTEMPT_WRONG[example.qid]
    else:
        answer = example.gold_answer

    ctx = " ".join(c.text for c in example.context)
    mem = " ".join(reflection_memory)
    stats = LLMStats(
        prompt_tokens=_estimate_tokens(example.question, ctx, mem),
        completion_tokens=_estimate_tokens(answer),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
    return answer, stats


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, LLMStats]:
    t0 = time.perf_counter()
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        judge = JudgeResult(
            score=1,
            reason="Final answer matches the gold answer after normalization.",
            missing_evidence=[],
            spurious_claims=[],
        )
    elif normalize_answer(answer) == "london":
        judge = JudgeResult(
            score=0,
            reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
            missing_evidence=["Need to identify the river that flows through London."],
            spurious_claims=[],
        )
    else:
        judge = JudgeResult(
            score=0,
            reason="The final answer selected the wrong second-hop entity.",
            missing_evidence=["Need to ground the answer in the second paragraph."],
            spurious_claims=[answer],
        )

    stats = LLMStats(
        prompt_tokens=_estimate_tokens(example.question, example.gold_answer, answer),
        completion_tokens=_estimate_tokens(judge.reason),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
    return judge, stats


def reflector(
    example: QAExample, attempt_id: int, judge: JudgeResult
) -> tuple[ReflectionEntry, LLMStats]:
    t0 = time.perf_counter()
    strategy = (
        "Do the second hop explicitly: birthplace city -> river through that city."
        if example.qid == "hp2"
        else "Verify the final entity against the second paragraph before answering."
    )
    entry = ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
        next_strategy=strategy,
    )
    stats = LLMStats(
        prompt_tokens=_estimate_tokens(example.question, judge.reason),
        completion_tokens=_estimate_tokens(entry.failure_reason, entry.lesson, entry.next_strategy),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
    return entry, stats
