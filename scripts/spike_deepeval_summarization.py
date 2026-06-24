"""Spike: is a better judging methodology more trustworthy than our absolute 0-1 score?

Runs ONE job (--job summarization|reasoning) on 2+ candidate models and grades the
SAME outputs three ways, so we can see whether the methodology — not the models —
was the problem:

  (a) CURRENT   — absolute 0-1 LLM-as-judge (what BlipShell does today)
  (b) PAIRWISE  — Arena-style A-vs-B, judged in BOTH orders to control position
                  bias (the upgrade the research recommends)
  (c) G-EVAL    — DeepEval's chain-of-thought rubric score (optional; skipped if
                  deepeval isn't installed or --geval-judge not passed)

It also prints mean output LENGTH per model: the research found verbosity alone
can swing absolute scores with no quality change, so if the absolute "winner" is
just the wordier model, that's the tell.

Pick the job deliberately: summarization is an EASY job where local often ties
cloud (so don't expect a gap); reasoning is a DISCRIMINATING job where cloud
should pull ahead — that's the one that proves the methodology can surface a real
gap. Run BOTH.

RUN ON THE OLLAMA PC (needs real models + a configured cloud judge), from repo dir:

    python scripts/spike_deepeval_summarization.py --job reasoning --models "qwen3:14b,minimax-m3:cloud"
    python scripts/spike_deepeval_summarization.py --job summarization --models "qwen3:14b,minimax-m3:cloud" --geval-judge qwen3:14b

The judge is whatever benchmark.judge_model/judge_endpoint point at in config.yaml.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from blipshell.benchmark.harness import BenchmarkHarness, _load_dataset, build_candidate_router
from blipshell.benchmark.judge import JudgeUnavailable, build_judge
from blipshell.core.config import ConfigManager
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.prompts import summarize_memory
from blipshell.llm.router import TaskType
from blipshell.models.config import get_ollama_url, resolve_env_vars

console = Console()

JOBS = {
    "summarization": {
        "item_header": "SOURCE MESSAGE",
        "pairwise_system": (
            "You are comparing two candidate memory-summaries of the same source message. "
            "A good summary is FAITHFUL (preserves names/numbers/key facts, invents nothing), "
            "CONCISE (1-2 short factual sentences), and a third-person factual note. Pick the "
            'better one; answer TIE only if truly equivalent. Answer ONLY "A", "B", or "TIE".'
        ),
    },
    "reasoning": {
        "item_header": "TASK",
        "pairwise_system": (
            "You are comparing two candidate answers to the same task. A good answer is CORRECT, "
            "COMPLETE (covers the key considerations), and ACTIONABLE (concrete, not vague). Pick "
            'the better one; answer TIE only if truly equivalent. Answer ONLY "A", "B", or "TIE".'
        ),
    },
}


def _words(text: str) -> int:
    return len(str(text).split())


async def _generate(job, model, provider, url, api_key, n):
    """Return (items, outputs) for one candidate model on the chosen job."""
    router = build_candidate_router(model, provider=provider, url=url, api_key=api_key)
    if job == "summarization":
        bm = _load_dataset("benchmark_models")
        items = [m["content"] for m in bm.TEST_MESSAGES if m.get("truth_rank", 1) >= 3][:n]
        outs = []
        for src in items:
            system, user = summarize_memory(src)
            try:
                outs.append(await router.generate(TaskType.SUMMARIZATION, user, system=system, think=False))
            except Exception as e:  # noqa: BLE001
                outs.append(f"ERROR: {e}")
        return items, outs
    # reasoning
    br = _load_dataset("benchmark_reasoning")
    items = [BenchmarkHarness._reasoning_task_text(t) for t in br.REASONING_TESTS][:n]
    results = await br.benchmark_reasoning(router)
    outs = [r.get("response", "") for r in results][:n]
    return items, outs


async def _absolute_scores(grade, items, outputs):
    """(a) Current method: mean absolute 0-1 judge score. grade(context, output)->0-1."""
    scores = []
    for ctx, out in zip(items, outputs):
        if str(out).startswith("ERROR:"):
            continue
        s = await grade(ctx, out)
        if s is not None:
            scores.append(s)
    return (sum(scores) / len(scores)) if scores else None


async def _pairwise_winrate(judge, items, a_outs, b_outs, header, system):
    """(b) Arena-style: judge A vs B per item in BOTH orders to cancel position bias."""
    a_wins = b_wins = ties = 0
    pos_first = pos_decided = 0  # position bias measured among DECIDED comparisons only

    async def _ask(item, first_out, second_out):
        prompt = (f"{header}:\n{item}\n\nANSWER A:\n{first_out}\n\nANSWER B:\n{second_out}\n\n"
                  'Which is better? Answer ONLY "A", "B", or "TIE".')
        try:
            raw = await judge._client.generate(prompt=prompt, model=judge.model, system=system, use_cache=False)
        except Exception:  # noqa: BLE001
            return None
        t = str(raw).strip().upper()
        return "first" if t.startswith("A") else "second" if t.startswith("B") else "tie" if t.startswith("TIE") else None

    for item, sa, sb in zip(items, a_outs, b_outs):
        if str(sa).startswith("ERROR:") or str(sb).startswith("ERROR:"):
            continue
        for r, a_first in ((await _ask(item, sa, sb), True), (await _ask(item, sb, sa), False)):
            if r is None:
                continue
            if r == "tie":
                ties += 1
                continue
            pos_decided += 1
            if r == "first":
                pos_first += 1
                (a_wins, b_wins) = (a_wins + 1, b_wins) if a_first else (a_wins, b_wins + 1)
            else:  # "second"
                (a_wins, b_wins) = (a_wins, b_wins + 1) if a_first else (a_wins + 1, b_wins)

    decided = a_wins + b_wins
    return {
        "a_winrate": (a_wins / decided) if decided else None,
        "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
        "tie_rate": ties / (decided + ties) if (decided + ties) else None,
        "position_first_rate": (pos_first / pos_decided) if pos_decided else None,
    }


def _geval_scores(job, items, outputs, judge_model, base_url):
    """(c) DeepEval G-Eval (optional). Mean score or None if unavailable."""
    try:
        from deepeval.metrics import GEval
        from deepeval.models import OllamaModel
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]G-Eval skipped: deepeval not available ({e}).[/yellow]")
        return None
    criteria = {
        "summarization": ("FAITHFUL to the source (names/numbers/key facts, invents nothing), "
                          "CONCISE (1-2 short factual sentences), third-person factual note."),
        "reasoning": ("CORRECT reasoning/conclusions for the task, COMPLETE (no major gaps), "
                      "ACTIONABLE (concrete, not vague hand-waving)."),
    }[job]
    judge = OllamaModel(model=judge_model, base_url=base_url, temperature=0)
    metric = GEval(name=f"{job} quality", criteria=criteria,
                   evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                   model=judge)
    scores = []
    for ctx, out in zip(items, outputs):
        if str(out).startswith("ERROR:"):
            continue
        try:
            metric.measure(LLMTestCase(input=ctx, actual_output=out))
            scores.append(metric.score)
        except Exception as e:  # noqa: BLE001
            console.print(f"[dim]G-Eval item failed: {e}[/dim]")
    return (sum(scores) / len(scores)) if scores else None


async def main():
    ap = argparse.ArgumentParser(description="Judging-methodology spike (summarization or reasoning)")
    ap.add_argument("--job", default="reasoning", choices=list(JOBS), help="Which job to test")
    ap.add_argument("--models", required=True, help="Comma-separated candidate models (2+)")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--url", default=None, help="Endpoint URL (openai provider)")
    ap.add_argument("--api-key-env", default=None, help="Env var with API key (openai provider)")
    ap.add_argument("--n", type=int, default=12, help="Number of items")
    ap.add_argument("--geval-judge", default=None, help="Local Ollama model for DeepEval G-Eval (default: skip)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 2:
        console.print("[red]Provide at least 2 models (--models a,b).[/red]")
        return

    config = ConfigManager(args.config).load()
    api_key = resolve_env_vars(f"${{{args.api_key_env}}}") if args.api_key_env else None
    url = args.url or (get_ollama_url(config.endpoints) if args.provider == "ollama" else None)
    if args.provider == "openai" and not url:
        console.print("[red]--provider openai requires --url[/red]")
        return

    try:
        judge = build_judge(config, EndpointManager(config.endpoints, config.llm))
    except JudgeUnavailable as e:
        console.print(f"[red]Judge unavailable: {e}.[/red]")
        return
    if judge is None:
        console.print("[red]No judge configured (benchmark.judge_model + judge_endpoint).[/red]")
        return

    job = args.job
    grade = judge.grade_summarization if job == "summarization" else judge.grade_reasoning
    console.print(f"[bold]Job: {job}[/bold]  |  Judge (absolute + pairwise): {judge.model}\n")

    outputs = {}
    items = None
    for m in models:
        console.print(f"[cyan]Generating {job} outputs: {m}[/cyan]")
        items, outputs[m] = await _generate(job, m, args.provider, url, api_key, args.n)
    console.print(f"[dim]{len(items)} items.[/dim]\n")

    rows = {}
    geval_base = url if args.provider == "ollama" else get_ollama_url(config.endpoints)
    for m in models:
        console.print(f"[dim]Absolute-scoring {m}...[/dim]")
        absolute = await _absolute_scores(grade, items, outputs[m])
        ok = [o for o in outputs[m] if not str(o).startswith("ERROR:")]
        mean_len = round(sum(_words(o) for o in ok) / len(ok), 1) if ok else None
        geval = None
        if args.geval_judge:
            console.print(f"[dim]G-Eval {m} (local judge {args.geval_judge})...[/dim]")
            geval = _geval_scores(job, items, outputs[m], args.geval_judge, geval_base)
        rows[m] = {"absolute": absolute, "geval": geval, "mean_len": mean_len}

    a, b = models[0], models[1]
    console.print(f"[dim]Pairwise {a} vs {b} (both orders)...[/dim]")
    pw = await _pairwise_winrate(judge, items, outputs[a], outputs[b], JOBS[job]["item_header"], JOBS[job]["pairwise_system"])

    # ---- report ----
    console.print()
    show_geval = any(rows[m]["geval"] is not None for m in models)
    t = Table(title=f"{job} - judging methodology comparison", show_lines=False)
    t.add_column("Model", style="cyan")
    t.add_column("Absolute 0-1 (current)", justify="right")
    if show_geval:
        t.add_column("G-Eval", justify="right")
    t.add_column("Mean length (words)", justify="right")
    for m in models:
        r = rows[m]
        row = [m, f"{r['absolute']:.3f}" if r["absolute"] is not None else "-"]
        if show_geval:
            row.append(f"{r['geval']:.3f}" if r["geval"] is not None else "-")
        row.append(f"{r['mean_len']}" if r["mean_len"] is not None else "-")
        t.add_row(*row)
    console.print(t)

    console.print()
    if pw["a_winrate"] is not None:
        console.print(f"[bold]Pairwise (order-controlled):[/bold] {a} won {pw['a_wins']}, "
                      f"{b} won {pw['b_wins']}, ties {pw['ties']}.")
        console.print(f"  {a} win-rate among decided: [bold]{pw['a_winrate']:.0%}[/bold]  "
                      f"(tie rate {pw['tie_rate']:.0%})")
        if pw["position_first_rate"] is not None:
            console.print(f"  [dim]Position-bias check (decided only): first-shown won "
                          f"{pw['position_first_rate']:.0%} (~50% = unbiased judge).[/dim]")
    else:
        console.print("[yellow]Pairwise produced no decided comparisons.[/yellow]")

    # ---- verdict ----
    console.print()
    abs_a, abs_b = rows[a]["absolute"], rows[b]["absolute"]
    if abs_a is not None and abs_b is not None and pw["a_winrate"] is not None:
        gap = abs(abs_a - abs_b)
        decisive = pw["a_winrate"] >= 0.65 or pw["a_winrate"] <= 0.35
        if gap < 0.05 and decisive:
            console.print("[bold green]VERDICT:[/bold green] absolute scores ~tied "
                          f"(delta {gap:.3f}) but pairwise is decisive - absolute scoring was hiding "
                          "the gap. Pairwise is the upgrade.")
        elif gap >= 0.05 and decisive:
            agree = (abs_a > abs_b) == (pw["a_winrate"] > 0.5)
            console.print(f"[bold]VERDICT:[/bold] both methods show a gap and "
                          f"{'AGREE' if agree else 'DISAGREE'} on the winner"
                          + ("." if agree else " - investigate."))
        else:
            console.print("[bold]VERDICT:[/bold] models look genuinely close on this job by both "
                          "methods. If this is the EASY job (summarization), run --job reasoning "
                          "to see whether a gap appears where it should.")
    console.print("[dim]If the higher absolute score is also the wordier model, suspect verbosity "
                  "bias - confirm against pairwise.[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
