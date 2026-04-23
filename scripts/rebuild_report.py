"""Rebuild report.{json,md} from existing JSONL records.

Useful after changes to:
- failure-mode classifier in agents.py (records get reclassified using their stored evaluator reason)
- failure_breakdown structure in reporting.py
- discussion / extensions in reporting.py

Does NOT call any LLM. Reads existing JSONL, reclassifies failure_mode, rewrites both
JSONL and report.

Run:
    python scripts/rebuild_report.py --out-dir outputs/real_run
"""
from __future__ import annotations
import sys
from pathlib import Path

# Allow `from src.reflexion_lab...` when invoking the script directly from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console

from src.reflexion_lab.agents import classify_failure
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.schemas import RunRecord
from src.reflexion_lab.utils import load_records_jsonl, normalize_answer, save_jsonl


def _detect_looping(rec: RunRecord) -> bool:
    """Match adaptive_max_attempts logic: a Reflexion record loops if a normalized
    answer appears more than once across its attempts."""
    if rec.agent_type != "reflexion" or len(rec.traces) < 2:
        return False
    norms = [normalize_answer(t.answer) for t in rec.traces]
    return len(norms) != len(set(norms))

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    out_dir: str = "outputs/real_run",
    dataset_name: str = "hotpot_100.json",
    mode: str = "openai",
) -> None:
    out_path = Path(out_dir)
    react_path = out_path / "react_runs.jsonl"
    refl_path = out_path / "reflexion_runs.jsonl"

    react_records = load_records_jsonl(react_path)
    reflexion_records = load_records_jsonl(refl_path)
    console.print(f"Loaded {len(react_records)} ReAct + {len(reflexion_records)} Reflexion records")

    fixed = 0
    for r in react_records + reflexion_records:
        if r.is_correct:
            new_mode = "none"
        elif _detect_looping(r):
            new_mode = "looping"
        else:
            new_mode = classify_failure(r.traces[-1].reason if r.traces else "")
        if r.failure_mode != new_mode:
            r.failure_mode = new_mode
            fixed += 1
    console.print(f"Reclassified {fixed} records")

    save_jsonl(react_path, react_records)
    save_jsonl(refl_path, reflexion_records)

    report = build_report(react_records + reflexion_records, dataset_name=dataset_name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    console.print(f"[green]Saved[/green] {json_path}")
    console.print(f"[green]Saved[/green] {md_path}")
    console.print(f"failure_modes: {list(report.failure_modes.keys())}")


if __name__ == "__main__":
    app()
