from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .backends import OllamaBackend, OpenAIBackend
from .backends.base import LLMBackend
from .cache import LLMCache
from .example import Example
from .metrics import METRICS
from .module import Module
from .optimizers import BootstrapFewShot, CombinedOptimizer, InstructionSearch
from .tracker import RunTracker

app = typer.Typer(help="PromptForge — automatic prompt optimization.", add_completion=False)
console = Console()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_task(path: str):
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Task file not found: {path}[/red]")
        raise typer.Exit(1)
    spec = importlib.util.spec_from_file_location("_pf_task", p)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "signature"):
        console.print("[red]task.py must define a `signature` variable.[/red]")
        raise typer.Exit(1)
    return mod


def _load_jsonl(path: str) -> list[Example]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(Example.from_dict(json.loads(line)))
    return examples


def _make_backend(backend: str, model: str, cache: LLMCache) -> LLMBackend:
    if backend == "openai":
        return OpenAIBackend(model=model, cache=cache)
    if backend == "ollama":
        return OllamaBackend(model=model, cache=cache)
    console.print(f"[red]Unknown backend: {backend}. Choose openai or ollama.[/red]")
    raise typer.Exit(1)


def _print_leaderboard(tracker: RunTracker, run_id: str, top_n: int = 10) -> None:
    rows = tracker.leaderboard(run_id, top_n=top_n)
    if not rows:
        console.print("[yellow]No results recorded for this run.[/yellow]")
        return

    table = Table(title=f"Leaderboard  run={run_id[:8]}", show_lines=True, expand=False)
    table.add_column("#", style="bold cyan", width=3, justify="right")
    table.add_column("Optimizer", style="cyan", no_wrap=True)
    table.add_column("Score", style="green", justify="right", width=7)
    table.add_column("Demos", justify="right", width=5)
    table.add_column("Instruction", max_width=70)

    for i, row in enumerate(rows, 1):
        demos = json.loads(row["demos_json"]) if row["demos_json"] else []
        table.add_row(
            str(i),
            row["optimizer"],
            f"{row['val_score']:.4f}",
            str(len(demos)),
            row["instruction"][:140],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

@app.command()
def optimize(
    task: str = typer.Option(..., "--task", "-t", help="Path to task.py"),
    train: str = typer.Option(..., "--train", help="Path to train.jsonl"),
    metric: str = typer.Option("exact_match", "--metric", "-m",
                                help="Metric: exact_match | f1 | embedding | llm_judge"),
    budget: int = typer.Option(30, "--budget", "-b", help="Max LLM call budget"),
    optimizer: str = typer.Option("combined", "--optimizer", "-o",
                                   help="bootstrap | instruction | combined"),
    backend: str = typer.Option("openai", "--backend", help="openai | ollama"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="Model name"),
    val_split: float = typer.Option(0.2, "--val-split", help="Fraction held out for validation"),
    k: int = typer.Option(3, "--k", help="Few-shot demos (BootstrapFewShot)"),
    n_proposals: int = typer.Option(5, "--n-proposals", help="Instruction candidates per round"),
    db: str = typer.Option("promptforge.db", "--db", help="SQLite tracker path"),
    cache_dir: str = typer.Option(".promptforge_cache", "--cache-dir"),
    output: Optional[str] = typer.Option(None, "--output", help="Save best prompt to JSON"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Resume an existing run"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Optimize a prompt using training examples and a chosen metric."""

    if metric not in METRICS:
        console.print(f"[red]Unknown metric '{metric}'. Available: {list(METRICS)}[/red]")
        raise typer.Exit(1)

    metric_fn = METRICS[metric]
    run_id = run_id or str(uuid.uuid4())

    console.rule(f"[bold]PromptForge optimize[/bold]  run={run_id[:8]}")
    console.print(f"  task={task}  metric={metric}  budget={budget}  optimizer={optimizer}")

    task_mod = _load_task(task)
    signature = task_mod.signature
    console.print(f"  signature: {signature}")

    examples = _load_jsonl(train)
    random.seed(seed)
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * val_split))
    val_ex = examples[:n_val]
    train_ex = examples[n_val:]
    console.print(f"  train={len(train_ex)}  val={len(val_ex)}  total={len(examples)}")

    cache = LLMCache(cache_dir)
    llm = _make_backend(backend, model, cache)
    tracker = RunTracker(db)
    module = Module(signature=signature, backend=llm)

    t0 = time.perf_counter()
    with console.status("[bold green]Optimizing…[/bold green]"):
        if optimizer == "bootstrap":
            best = BootstrapFewShot(k=k, tracker=tracker).optimize(
                module, train_ex, metric_fn, run_id, budget, val_ex
            )
        elif optimizer == "instruction":
            best = InstructionSearch(n_proposals=n_proposals, tracker=tracker).optimize(
                module, val_ex, metric_fn, run_id, budget
            )
        elif optimizer == "combined":
            best = CombinedOptimizer(k=k, n_proposals=n_proposals, tracker=tracker).optimize(
                module, train_ex, val_ex, metric_fn, run_id, budget
            )
        else:
            console.print(f"[red]Unknown optimizer: {optimizer}[/red]")
            raise typer.Exit(1)

    elapsed = time.perf_counter() - t0
    console.print(
        f"\n[green]Done in {elapsed:.1f}s[/green]"
        f"  api_calls={llm.call_count}  cache_hits={llm.cache_hits}"
    )

    _print_leaderboard(tracker, run_id)

    best_row = tracker.best(run_id)
    if best_row:
        console.print(f"\n[bold]Best instruction[/bold] (val={best_row['val_score']:.4f}):")
        console.print(f"  [italic]{best_row['instruction']}[/italic]")

    if output:
        # Use the highest-scoring DB entry, not the optimizer's final state, which
        # may be worse if a later phase degraded the score.
        if best_row:
            saved_demos = json.loads(best_row["demos_json"]) if best_row["demos_json"] else []
            result = {
                "run_id": run_id,
                "optimizer": optimizer,
                "metric": metric,
                "val_score": best_row["val_score"],
                "instruction": best_row["instruction"],
                "demos": saved_demos,
            }
        else:
            result = {
                "run_id": run_id,
                "optimizer": optimizer,
                "metric": metric,
                "val_score": 0.0,
                "instruction": best.instruction,
                "demos": [ex.to_dict() for ex in best.demos],
            }
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        console.print(f"\nSaved to [cyan]{output}[/cyan]")

    console.print(f"\n[dim]run_id={run_id}[/dim]")


@app.command()
def eval(  # noqa: A001
    task: str = typer.Option(..., "--task", "-t", help="Path to task.py"),
    test: str = typer.Option(..., "--test", help="Path to test.jsonl"),
    metric: str = typer.Option("exact_match", "--metric", "-m"),
    backend: str = typer.Option("openai", "--backend"),
    model: str = typer.Option("gpt-4o-mini", "--model"),
    prompt_file: Optional[str] = typer.Option(None, "--prompt-file", "-p",
                                               help="JSON from optimize --output"),
    cache_dir: str = typer.Option(".promptforge_cache", "--cache-dir"),
    show_errors: bool = typer.Option(False, "--show-errors", help="Print wrong predictions"),
) -> None:
    """Evaluate a (optionally optimized) prompt on a held-out test set."""

    if metric not in METRICS:
        console.print(f"[red]Unknown metric: {metric}[/red]")
        raise typer.Exit(1)

    metric_fn = METRICS[metric]
    task_mod = _load_task(task)
    signature = task_mod.signature
    test_examples = _load_jsonl(test)
    cache = LLMCache(cache_dir)
    llm = _make_backend(backend, model, cache)

    instruction: str | None = None
    demos: list[Example] = []

    if prompt_file:
        data = json.loads(Path(prompt_file).read_text())
        instruction = data.get("instruction")
        demos = [Example.from_dict(d) for d in data.get("demos", [])]
        console.print(f"Loaded optimized prompt from [cyan]{prompt_file}[/cyan]")

    module = Module(signature=signature, backend=llm, demos=demos, instruction=instruction)
    output_field = signature.outputs[0].name

    scores: list[float] = []
    errors: list[tuple] = []

    with console.status("[bold green]Evaluating…[/bold green]"):
        for ex in test_examples:
            try:
                pred = module(**ex.inputs)
                pred_val = pred.get(output_field, "")
                gold_val = ex.outputs.get(output_field, "")
                score = metric_fn(pred_val, gold_val)
                scores.append(score)
                if score < 1.0:
                    errors.append((ex, pred_val, score))
            except Exception as exc:
                scores.append(0.0)
                errors.append((ex, str(exc), 0.0))

    avg = sum(scores) / len(scores) if scores else 0.0

    table = Table(title="Eval Results", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Score", style="green", justify="right")
    table.add_column("N examples", justify="right")
    table.add_row(metric, f"{avg:.4f}", str(len(scores)))
    console.print(table)

    if show_errors and errors:
        console.print(f"\n[yellow]Misses ({len(errors)}):[/yellow]")
        for ex, pred_val, sc in errors[:15]:
            console.print(
                f"  inputs={ex.inputs}  pred={pred_val!r:30}  "
                f"gold={ex.outputs}  score={sc:.2f}"
            )

    console.print(f"\napi_calls={llm.call_count}  cache_hits={llm.cache_hits}")


@app.command()
def leaderboard(
    db: str = typer.Option("promptforge.db", "--db"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    top_n: int = typer.Option(10, "--top-n"),
) -> None:
    """Print the leaderboard for an optimization run."""
    tracker = RunTracker(db)

    if run_id is None:
        runs = tracker.all_runs()
        if not runs:
            console.print("[yellow]No runs found in database.[/yellow]")
            raise typer.Exit(0)
        run_id = runs[0]
        console.print(f"Most recent run: [bold]{run_id}[/bold]")

    _print_leaderboard(tracker, run_id, top_n=top_n)
