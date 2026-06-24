"""Spike: is a better judging methodology more trustworthy than our absolute 0-1 score?

Runs ONE job (summarization) on 2+ candidate models and grades the SAME outputs
three ways, so we can see whether the methodology — not the models — was the
problem:

  (a) CURRENT   — absolute 0-1 LLM-as-judge (what BlipShell does today)
  (b) PAIRWISE  — Arena-style A-vs-B, judged in BOTH orders to control position
                  bias (the upgrade the research recommends)
  (c) G-EVAL    — DeepEval's chain-of-thought rubric score (optional; skipped if
                  deepeval isn't installed)

It also prints mean summary LENGTH per model, because the research found verbosity
alone can swing absolute scores with no quality change — if the "winner" on
absolute score is just the wordier model, that's the tell.

Verdict line: if absolute scores are ~tied but pairwise (consistent across both
orders) clearly favors one model, the absolute method was hiding the real gap.

RUN ON THE OLLAMA PC (needs real models + a configured cloud judge), from the
repo dir:

    python scripts/spike_deepeval_summarization.py --models "qwen3:14b,minimax-m3:cloud"
    # cloud candidate:
    python scripts/spike_deepeval_summarization.py \
        --models "minimax/minimax-m3" --provider openai \
        --url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY

The judge is whatever benchmark.judge_model/judge_endpoint point at in config.yaml.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from blipshell.benchmark.harness import _load_dataset, build_candidate_router
from blipshell.benchmark.judge import JudgeUnavailable, build_judge
from blipshell.core.config import ConfigManager
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.prompts import summarize_memory
from blipshell.llm.router import TaskType
from blipshell.models.config import get_ollama_url, resolve_env_vars

console = Console()


def _words(text: str) -> int:
    return len(str(text).split())


async def _generate_summaries(model, provider, url, api_key, sources):
    """Run each source through one candidate model's summarization path."""
    router = build_candidate_router(model, provider=provider, url=url, api_key=api_key)
    out = []
    for src in sources:
        system, user = summarize_memory(src)
        try:
            summ = await router.generate(TaskType.SUMMARIZATION, user, system=system, think=False)
        except Exception as e:  # noqa: BLE001
            summ = f"ERROR: {e}"
        out.append(summ)
    return out


async def _absolute_scores(judge, sources, summaries):
    """(a) Current method: mean absolute 0-1 judge score."""
    scores = []
    for src, summ in zip(sources, summaries):
        if str(summ).startswith("ERROR:"):
            continue
        s = await judge.grade_summarization(src, summ)
        if s is not None:
            scores.append(s)
    return (sum(scores) / len(scores)) if scores else None


_PAIRWISE_SYSTEM = (
    "You are comparing two candidate memory-summaries of the same source message. "
    "A good memory summary is FAITHFUL (preserves names/numbers/key facts, invents "
    "nothing), CONCISE (1-2 short factual sentences), and written as a third-person "
    "factual note. Pick the better summary. If they are equivalent, answer TIE. "
    'Answer with ONLY one token: "A", "B", or "TIE".'
)


async def _pairwise_winrate(judge, sources, a_summaries, b_summaries, label_a, label_b):
    """(b) Arena-style: judge A vs B for each source in BOTH orders to cancel
    position bias. Returns (a_winrate, position_bias_rate, ties)."""
    a_wins = b_wins = ties = comparisons = 0
    position_first_wins = 0  # how often the FIRST-shown option won (bias signal)
    position_total = 0

    async def _ask(first_text, second_text):
        prompt = (
            f"SOURCE MESSAGE:\n{first_text[0]}\n\n"
            f"SUMMARY A:\n{first_text[1]}\n\nSUMMARY B:\n{second_text}\n\n"
            'Which summary is better? Answer ONLY "A", "B", or "TIE".'
        )
        try:
            raw = await judge._client.generate(
                prompt=prompt, model=judge.model, system=_PAIRWISE_SYSTEM, use_cache=False,
            )
        except Exception:  # noqa: BLE001
            return None
        t = str(raw).strip().upper()
        if t.startswith("A"):
            return "first"
        if t.startswith("B"):
            return "second"
        if t.startswith("TIE"):
            return "tie"
        return None

    for src, sa, sb in zip(sources, a_summaries, b_summaries):
        if str(sa).startswith("ERROR:") or str(sb).startswith("ERROR:"):
            continue
        # Order 1: A shown first
        r1 = await _ask((src, sa), sb)
        # Order 2: B shown first
        r2 = await _ask((src, sb), sa)
        for r, a_is_first in ((r1, True), (r2, False)):
            if r is None:
                continue
            position_total += 1
            if r == "first":
                position_first_wins += 1
                winner = "A" if a_is_first else "B"
            elif r == "second":
                winner = "B" if a_is_first else "A"
            else:
                winner = "TIE"
            comparisons += 1
            if winner == "A":
                a_wins += 1
            elif winner == "B":
                b_wins += 1
            else:
                ties += 1

    decided = a_wins + b_wins
    a_winrate = (a_wins / decided) if decided else None
    pos_bias = (position_first_wins / position_total) if position_total else None
    return {
        "a_winrate": a_winrate, "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
        "comparisons": comparisons, "position_first_rate": pos_bias,
        "label_a": label_a, "label_b": label_b,
    }


def _geval_scores(sources, summaries, judge_model, base_url):
    """(c) DeepEval G-Eval (optional). Returns mean score or None if unavailable."""
    try:
        from deepeval.metrics import GEval
        from deepeval.models import OllamaModel
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]G-Eval skipped: deepeval not installed ({e}). "
                      "pip install deepeval to include it.[/yellow]")
        return None

    judge = OllamaModel(model=judge_model, base_url=base_url, temperature=0)
    metric = GEval(
        name="Memory summary quality",
        criteria=("The summary is FAITHFUL to the source (preserves names, numbers, key "
                  "facts; invents nothing), CONCISE (1-2 short factual sentences), and a "
                  "third-person factual note (no 'User asked'/'Assistant said', no markup)."),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
    )
    scores = []
    for src, summ in zip(sources, summaries):
        if str(summ).startswith("ERROR:"):
            continue
        try:
            metric.measure(LLMTestCase(input=src, actual_output=summ))
            scores.append(metric.score)
        except Exception as e:  # noqa: BLE001
            console.print(f"[dim]G-Eval item failed: {e}[/dim]")
    return (sum(scores) / len(scores)) if scores else None


async def main():
    ap = argparse.ArgumentParser(description="Summarization judging-methodology spike")
    ap.add_argument("--models", required=True, help="Comma-separated candidate models (2+)")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--url", default=None, help="Endpoint URL (openai provider)")
    ap.add_argument("--api-key-env", default=None, help="Env var with API key (openai provider)")
    ap.add_argument("--n", type=int, default=12, help="Number of source messages to summarize")
    ap.add_argument("--geval-judge", default=None,
                    help="Local Ollama model for DeepEval G-Eval (default: skip G-Eval)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 2:
        console.print("[red]Provide at least 2 models to compare (--models a,b).[/red]")
        return

    config = ConfigManager(args.config).load()
    api_key = resolve_env_vars(f"${{{args.api_key_env}}}") if args.api_key_env else None
    url = args.url or (get_ollama_url(config.endpoints) if args.provider == "ollama" else None)
    if args.provider == "openai" and not url:
        console.print("[red]--provider openai requires --url[/red]")
        return

    # Judge from config (cloud). Used for both absolute and pairwise (same judge =
    # apples-to-apples between the two methods).
    try:
        judge = build_judge(config, EndpointManager(config.endpoints, config.llm))
    except JudgeUnavailable as e:
        judge = None
        console.print(f"[red]Judge unavailable: {e}. Set benchmark.judge_model/endpoint.[/red]")
        return
    if judge is None:
        console.print("[red]No judge configured. Set benchmark.judge_model + judge_endpoint.[/red]")
        return
    console.print(f"[dim]Judge (absolute + pairwise): {judge.model}[/dim]")

    # Substantive sources only (skip filler — nothing to summarize there).
    bm = _load_dataset("benchmark_models")
    sources = [m["content"] for m in bm.TEST_MESSAGES if m.get("truth_rank", 1) >= 3][: args.n]
    console.print(f"[dim]{len(sources)} source messages.[/dim]\n")

    # Generate summaries per model.
    summaries = {}
    for m in models:
        console.print(f"[cyan]Generating summaries: {m}[/cyan]")
        summaries[m] = await _generate_summaries(m, args.provider, url, api_key, sources)

    # (a) absolute, (c) g-eval, length per model
    rows = {}
    for m in models:
        console.print(f"[dim]Absolute-scoring {m}…[/dim]")
        absolute = await _absolute_scores(judge, sources, summaries[m])
        ok = [s for s in summaries[m] if not str(s).startswith("ERROR:")]
        mean_len = round(sum(_words(s) for s in ok) / len(ok), 1) if ok else None
        geval = None
        if args.geval_judge:
            console.print(f"[dim]G-Eval {m} (local judge {args.geval_judge})…[/dim]")
            geval = _geval_scores(sources, summaries[m], args.geval_judge, url if args.provider == "ollama" else get_ollama_url(config.endpoints))
        rows[m] = {"absolute": absolute, "geval": geval, "mean_len": mean_len}

    # (b) pairwise for the first two models
    a, b = models[0], models[1]
    console.print(f"[dim]Pairwise {a} vs {b} (both orders)…[/dim]")
    pw = await _pairwise_winrate(judge, sources, summaries[a], summaries[b], a, b)

    # ---- report ----
    console.print()
    t = Table(title="Summarization — judging methodology comparison", show_lines=False)
    t.add_column("Model", style="cyan")
    t.add_column("Absolute 0-1 (current)", justify="right")
    if any(rows[m]["geval"] is not None for m in models):
        t.add_column("G-Eval (DeepEval)", justify="right")
    t.add_column("Mean length (words)", justify="right")
    show_geval = any(rows[m]["geval"] is not None for m in models)
    for m in models:
        r = rows[m]
        row = [m, f"{r['absolute']:.3f}" if r["absolute"] is not None else "—"]
        if show_geval:
            row.append(f"{r['geval']:.3f}" if r["geval"] is not None else "—")
        row.append(f"{r['mean_len']}" if r["mean_len"] is not None else "—")
        t.add_row(*row)
    console.print(t)

    console.print()
    if pw["a_winrate"] is not None:
        console.print(f"[bold]Pairwise (order-controlled):[/bold] {a} won "
                      f"{pw['a_wins']}, {b} won {pw['b_wins']}, ties {pw['ties']} "
                      f"({pw['comparisons']} comparisons across both orders).")
        console.print(f"  {a} win-rate among decided: [bold]{pw['a_winrate']:.0%}[/bold]")
        if pw["position_first_rate"] is not None:
            console.print(f"  [dim]Position-bias check: first-shown option won "
                          f"{pw['position_first_rate']:.0%} of the time (≈50% = unbiased).[/dim]")
    else:
        console.print("[yellow]Pairwise produced no decided comparisons.[/yellow]")

    # ---- verdict ----
    console.print()
    abs_a, abs_b = rows[a]["absolute"], rows[b]["absolute"]
    if abs_a is not None and abs_b is not None and pw["a_winrate"] is not None:
        abs_gap = abs(abs_a - abs_b)
        pw_decisive = pw["a_winrate"] >= 0.65 or pw["a_winrate"] <= 0.35
        if abs_gap < 0.05 and pw_decisive:
            console.print("[bold green]VERDICT:[/bold green] absolute scores were ~tied "
                          f"(Δ {abs_gap:.3f}) but pairwise clearly favors one model — the "
                          "absolute method was hiding the real gap. Pairwise is the upgrade.")
        elif abs_gap >= 0.05 and pw_decisive:
            agree = (abs_a > abs_b) == (pw["a_winrate"] > 0.5)
            console.print(f"[bold]VERDICT:[/bold] both methods show a gap and "
                          f"{'agree' if agree else 'DISAGREE'} on the winner.")
        else:
            console.print("[bold]VERDICT:[/bold] models look genuinely close on this job "
                          "by both methods; check the length column for verbosity effects.")
    console.print("[dim]If the higher absolute score is also the wordier model, suspect "
                  "verbosity bias — confirm with the pairwise result.[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
