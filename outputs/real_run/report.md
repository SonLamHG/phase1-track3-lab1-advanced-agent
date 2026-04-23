# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: openai
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.88 | 0.95 | 0.07 |
| Avg attempts | 1 | 1.17 | 0.17 |
| Avg token estimate | 2278.26 | 3042.16 | 763.9 |
| Avg latency (ms) | 3292.7 | 5039.93 | 1747.23 |

## Failure modes
```json
{
  "none": {
    "react": 88,
    "reflexion": 95
  },
  "wrong_final_answer": {
    "react": 7
  },
  "incomplete_multi_hop": {
    "react": 1
  },
  "entity_drift": {
    "react": 4
  },
  "looping": {
    "reflexion": 5
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding
- adaptive_max_attempts

## Discussion
Headline: Reflexion improved exact-match from 88.0% to 95.0% (+7.0pp) on 100 HotpotQA-distractor questions, paying +34% more tokens and +53% more latency per question. Cost-effectiveness: only 14% of questions triggered a reflection (avg attempts = 1.17); the remaining 86% were answered correctly on the first attempt and incurred zero overhead, so the headline overhead is amortised over a small minority of hard cases. Where reflection helped: ReAct failure modes [wrong_final_answer=7, entity_drift=4, incomplete_multi_hop=1] became [looping=5] under Reflexion, fixing -7 wrong_final_answer, -4 entity_drift, -1 incomplete_multi_hop; this matches the intuition that an explicit critique step rescues entity-drift and incomplete-multi-hop answers when the model can re-read the context with a concrete next_strategy. The adaptive_max_attempts extension stopped 5 Reflexion runs early after detecting a duplicated wrong answer (looping), saving ~5 extra reflection cycles without harming EM. Caveats: (1) the failure-mode classifier is keyword-heuristic over the evaluator's free-text reason and may misroute borderline cases; (2) running the openai backend at low reasoning effort is non-deterministic, so re-runs vary by a few percentage points (EM swung between 0.95 and 0.99 across two consecutive runs of this same setup); (3) actor and evaluator share the same model, which could mask blind spots — using a stronger evaluator would tighten the EM measurement, and a confidence-based variant of adaptive_max_attempts (currently looping-based only) is the most natural next step.
