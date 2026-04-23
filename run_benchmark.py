from __future__ import annotations

# IMPORTANT: load .env BEFORE importing src.reflexion_lab.agents — that module binds the
# runtime backend (mock vs openai) at import time by reading os.environ["LLM_PROVIDER"].
from dotenv import load_dotenv
load_dotenv()

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.schemas import QAExample, RunRecord
from src.reflexion_lab.utils import append_jsonl, load_dataset, load_records_jsonl

console = Console()
app = typer.Typer(add_completion=False)


# Approx pricing per 1M tokens (USD). Used only for the live cost estimate
# in the progress bar — actual billing is on the OpenAI dashboard.
PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-5-nano":  (0.05, 0.40),
    "gpt-5-mini":  (0.25, 2.00),
    "gpt-5":       (1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o":      (2.50, 10.00),
    "o4-mini":     (1.10, 4.40),
}


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def estimate_cost(total_tokens: int, model: str) -> float:
    """Blended cost estimate. Reasoning models tend to spend more on output (hidden reasoning)."""
    if model not in PRICING_PER_1M:
        return 0.0
    in_rate, out_rate = PRICING_PER_1M[model]
    in_share = 0.70 if _is_reasoning_model(model) else 0.85
    blended_rate = in_share * in_rate + (1 - in_share) * out_rate
    return total_tokens / 1_000_000 * blended_rate


def fmt_diffs(correct: dict[str, int], total: dict[str, int]) -> str:
    parts = []
    for d in ("easy", "medium", "hard"):
        if total[d] > 0:
            parts.append(f"{d[0].upper()}{correct[d]:>2}/{total[d]:<2}")
    return " ".join(parts) if parts else "-"


def run_with_progress(
    agent,
    examples: list[QAExample],
    jsonl_path: Path,
    label: str,
    model: str,
) -> list[RunRecord]:
    existing = load_records_jsonl(jsonl_path)
    done_qids = {r.qid for r in existing}
    remaining = [e for e in examples if e.qid not in done_qids]
    if existing:
        console.print(
            f"[yellow]{label}: resuming — {len(existing)} done, {len(remaining)} remaining[/yellow]"
        )

    ex_by_qid = {e.qid: e for e in examples}
    correct = sum(1 for r in existing if r.is_correct)
    tokens = sum(r.token_estimate for r in existing)
    diff_total: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    diff_correct: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for r in existing:
        d = ex_by_qid[r.qid].difficulty
        diff_total[d] += 1
        if r.is_correct:
            diff_correct[d] += 1

    records = list(existing)

    with Progress(
        TextColumn(f"[bold cyan]{label:9}[/bold cyan]"),
        BarColumn(bar_width=24),
        TextColumn("{task.completed:>3}/{task.total:<3}"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        TextColumn("EM={task.fields[em]:>5.1%}"),
        TextColumn("[{task.fields[diffs]}]"),
        TextColumn("tok={task.fields[tokens]:>7,}"),
        TextColumn("~${task.fields[cost]:>5.3f}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            "",
            total=len(examples),
            completed=len(existing),
            em=correct / max(1, len(existing)),
            diffs=fmt_diffs(diff_correct, diff_total),
            tokens=tokens,
            cost=estimate_cost(tokens, model),
        )
        for ex in remaining:
            try:
                r = agent.run(ex)
            except Exception as e:
                console.print(
                    f"[red]ERROR on {ex.qid}: {type(e).__name__}: {e}[/red]"
                )
                raise
            records.append(r)
            append_jsonl(jsonl_path, r)

            correct += int(r.is_correct)
            tokens += r.token_estimate
            diff_total[ex.difficulty] += 1
            if r.is_correct:
                diff_correct[ex.difficulty] += 1

            progress.update(
                task,
                advance=1,
                em=correct / len(records),
                diffs=fmt_diffs(diff_correct, diff_total),
                tokens=tokens,
                cost=estimate_cost(tokens, model),
            )

    return records


@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    fresh: bool = typer.Option(
        False, "--fresh", help="Delete existing JSONL files and start over (default: resume)."
    ),
) -> None:
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()
    model = (
        os.environ.get("LLM_ACTOR_MODEL", "gpt-4o-mini")
        if provider == "openai"
        else "mock"
    )
    console.print(f"[cyan]Provider[/cyan]: {provider} ({model})")
    examples = load_dataset(dataset)
    console.print(f"[cyan]Dataset [/cyan]: {dataset} ({len(examples)} examples)")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    react_jsonl = out_path / "react_runs.jsonl"
    reflexion_jsonl = out_path / "reflexion_runs.jsonl"

    if fresh:
        for p in (react_jsonl, reflexion_jsonl):
            if p.exists():
                p.unlink()
                console.print(f"[yellow]Deleted {p}[/yellow]")

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)

    react_records = run_with_progress(react, examples, react_jsonl, "ReAct", model)
    reflexion_records = run_with_progress(
        reflexion, examples, reflexion_jsonl, "Reflexion", model
    )

    all_records = react_records + reflexion_records
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=provider)
    json_path, md_path = save_report(report, out_path)
    console.print(f"[green]Saved[/green] {json_path}")
    console.print(f"[green]Saved[/green] {md_path}")
    console.print_json(json.dumps(report.summary))


if __name__ == "__main__":
    app()
