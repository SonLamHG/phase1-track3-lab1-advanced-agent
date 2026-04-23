"""Prepare a balanced 100-sample HotpotQA dataset (distractor variant).

- Source: HuggingFace `hotpotqa/hotpot_qa`, config `distractor`, split `train`
  (the official validation split is all `hard`, so train is used to get a
  balanced easy/medium/hard distribution)
- Distribution: 34 easy / 33 medium / 33 hard (deterministic via SEED)
- Keeps ALL 10 paragraphs per sample (2 gold + 8 distractors)
- Output schema matches `src/reflexion_lab/schemas.QAExample`

Run:
    pip install datasets
    python scripts/prepare_hotpot_100.py
"""
from __future__ import annotations
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "hotpot_100.json"
SEED = 42
TARGETS = {"easy": 34, "medium": 33, "hard": 33}


def convert_context(ctx: dict) -> list[dict]:
    # HotpotQA stores context as parallel lists: {'title': [...], 'sentences': [[...], ...]}.
    return [
        {"title": title, "text": " ".join(sents).strip()}
        for title, sents in zip(ctx["title"], ctx["sentences"])
    ]


def main() -> None:
    rng = random.Random(SEED)
    print("Loading HotpotQA distractor train split ...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

    buckets: dict[str, list] = {lvl: [] for lvl in TARGETS}
    for row in ds:
        lvl = row.get("level")
        if lvl in buckets:
            buckets[lvl].append(row)

    for lvl, need in TARGETS.items():
        if len(buckets[lvl]) < need:
            raise RuntimeError(f"Not enough {lvl} samples: have {len(buckets[lvl])}, need {need}")

    chosen: list = []
    for lvl, need in TARGETS.items():
        rng.shuffle(buckets[lvl])
        chosen.extend(buckets[lvl][:need])
    rng.shuffle(chosen)

    out = []
    for i, row in enumerate(chosen, start=1):
        out.append({
            "qid": f"hp{i:03d}",
            "difficulty": row["level"],
            "question": row["question"].strip(),
            "gold_answer": row["answer"].strip(),
            "context": convert_context(row["context"]),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    dist = {lvl: sum(1 for o in out if o["difficulty"] == lvl) for lvl in TARGETS}
    avg_ctx = sum(len(o["context"]) for o in out) / len(out)
    print(f"Wrote {len(out)} samples to {OUT_PATH}")
    print(f"Distribution: {dist}")
    print(f"Avg paragraphs/sample: {avg_ctx:.1f}")


if __name__ == "__main__":
    main()
