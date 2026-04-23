from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable
from .schemas import QAExample, RunRecord

def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def load_dataset(path: str | Path) -> list[QAExample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [QAExample.model_validate(item) for item in raw]

def save_jsonl(path: str | Path, records: Iterable[RunRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def append_jsonl(path: str | Path, record: RunRecord) -> None:
    """Append a single record. Used for streaming progress so a crash mid-run keeps prior data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def load_records_jsonl(path: str | Path) -> list[RunRecord]:
    """Load previously-streamed records (for --resume). Returns [] if file missing."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[RunRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(RunRecord.model_validate_json(line))
    return out
