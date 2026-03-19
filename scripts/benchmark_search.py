"""Standalone search quality benchmark — no Agent, no Router, no LLM calls.

Connects directly to SQLite + ChromaDB to measure search precision, recall,
and timing. Designed for A/B testing search config changes.

Usage:
    # Discover mode — prints top results for manual ground truth creation
    python scripts/benchmark_search.py --discover
    python scripts/benchmark_search.py --discover --db data/blipshell.db --chroma data/chroma

    # Benchmark mode — measures against ground truth
    python scripts/benchmark_search.py
    python scripts/benchmark_search.py --ground-truth data/search_ground_truth.json

    # A/B comparison — override search config for the B run
    python scripts/benchmark_search.py --ab --similarity-threshold 0.25

    # Custom queries file
    python scripts/benchmark_search.py --queries data/my_queries.json

Options:
    --db PATH              SQLite database path (default: data/blipshell.db)
    --chroma PATH          ChromaDB directory (default: data/chroma)
    --ollama URL           Ollama URL for embeddings (default: http://localhost:11434)
    --output PATH          Output JSON report path
    --discover             Print top results per query (for ground truth creation)
    --ground-truth PATH    Ground truth file for precision/recall scoring
    --ab                   Run twice: baseline config then overridden config
    --project NAME         Scope search to a project
    --quiet                JSON only, no Rich output

Search config overrides (for A/B testing):
    --similarity-threshold FLOAT
    --min-importance FLOAT
    --entity-cap INT
    --fts-weight FLOAT
    --recency-boost FLOAT
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Default test queries (used when no --queries file provided)
# ---------------------------------------------------------------------------

DEFAULT_QUERIES = [
    # Personal facts
    {"query": "what is my name", "category": "personal"},
    {"query": "where do I work", "category": "personal"},
    {"query": "what programming languages do I know", "category": "personal"},
    {"query": "what computer setup do I have", "category": "personal"},
    {"query": "what are my preferences", "category": "personal"},

    # Project context
    {"query": "how does entity extraction work", "category": "project"},
    {"query": "what model do we use for ranking", "category": "project"},
    {"query": "how does the search pipeline work", "category": "project"},
    {"query": "what is the executor architecture", "category": "project"},
    {"query": "how does the memory system work", "category": "project"},

    # Temporal / recent
    {"query": "what did we work on recently", "category": "temporal"},
    {"query": "what was the last bug we fixed", "category": "temporal"},
    {"query": "what changes did we make to the config", "category": "temporal"},
    {"query": "what tests did we run", "category": "temporal"},
    {"query": "what models have we benchmarked", "category": "temporal"},

    # Foundational
    {"query": "what is BlipShell", "category": "foundational"},
    {"query": "what is the overall architecture", "category": "foundational"},
    {"query": "what databases does the system use", "category": "foundational"},
    {"query": "how do sessions work", "category": "foundational"},
    {"query": "what is the memory processing pipeline", "category": "foundational"},

    # Keyword-dependent (need FTS to work well)
    {"query": "OllamaGate contention", "category": "keyword"},
    {"query": "ChromaDB sync drift", "category": "keyword"},
    {"query": "task_complete tool", "category": "keyword"},
    {"query": "PII sanitization Presidio", "category": "keyword"},
    {"query": "centroid tagger", "category": "keyword"},
]


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Result of running one search query."""
    query: str
    category: str
    results: list[dict] = field(default_factory=list)  # [{id, summary, similarity, boosted_score, ...}]
    search_stats: dict = field(default_factory=dict)
    timing_ms: float = 0.0
    # Precision/recall (only when ground truth available)
    expected_ids: list[int] = field(default_factory=list)
    precision_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    timestamp: str = ""
    db_path: str = ""
    chroma_path: str = ""
    config_label: str = "baseline"
    query_results: list[QueryResult] = field(default_factory=list)
    # Aggregates
    avg_precision_at_5: float = 0.0
    avg_recall_at_10: float = 0.0
    avg_mrr: float = 0.0
    avg_timing_ms: float = 0.0
    total_fts_hits: int = 0
    total_chroma_hits: int = 0
    total_entity_hits: int = 0
    total_queries: int = 0


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def run_benchmark(
    db_path: str,
    chroma_path: str,
    ollama_url: str,
    queries: list[dict],
    ground_truth: dict | None = None,
    project: str | None = None,
    config_overrides: dict | None = None,
    config_label: str = "baseline",
) -> BenchmarkReport:
    """Run the search benchmark against the real database."""

    # Lazy imports — ChromaDB may not be available on dev machine
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.search import MemorySearch

    # Initialize stores
    sqlite = SQLiteStore(db_path)
    await sqlite.initialize()

    chroma = ChromaStore(
        persist_dir=chroma_path,
        ollama_url=ollama_url,
    )
    chroma.initialize()

    # Create search with no router (not used in search())
    search = MemorySearch(sqlite=sqlite, chroma=chroma, router=None)

    # Apply config overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(search, key):
                setattr(search, key, value)

    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        db_path=db_path,
        chroma_path=chroma_path,
        config_label=config_label,
        total_queries=len(queries),
    )

    for q in queries:
        query_text = q["query"]
        category = q.get("category", "unknown")
        expected_ids = []
        if ground_truth and query_text in ground_truth:
            expected_ids = ground_truth[query_text]

        # Run search
        start = time.monotonic()
        try:
            results = await search.search(
                query=query_text,
                n_results=15,
                active_project=project,
            )
        except Exception as e:
            console.print(f"[red]Search failed for '{query_text}': {e}[/red]")
            results = []
        elapsed_ms = (time.monotonic() - start) * 1000

        # Also search lessons
        try:
            lessons = await search.search_lessons(
                query=query_text,
                n_results=5,
                active_project=project,
            )
        except Exception:
            lessons = []

        stats = search.last_search_stats or {}

        # Build result list
        result_list = []
        for r in results:
            result_list.append({
                "id": r.memory_id,
                "summary": r.summary[:200],
                "similarity": round(r.similarity, 4),
                "boosted_score": round(r.boosted_score, 4),
                "importance": round(r.importance, 4),
                "rank": r.rank,
                "tags": r.tags[:5] if r.tags else [],
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            })

        # Compute precision/recall/MRR
        returned_ids = [r.memory_id for r in results]
        p_at_5 = 0.0
        r_at_10 = 0.0
        mrr = 0.0

        if expected_ids:
            expected_set = set(expected_ids)
            # Precision@5: of the top 5, how many are expected?
            top_5 = returned_ids[:5]
            p_at_5 = len(set(top_5) & expected_set) / min(5, len(expected_set)) if expected_set else 0.0
            # Recall@10: of all expected, how many appear in top 10?
            top_10 = returned_ids[:10]
            r_at_10 = len(set(top_10) & expected_set) / len(expected_set) if expected_set else 0.0
            # MRR: reciprocal of the rank of the first expected result
            for i, rid in enumerate(returned_ids):
                if rid in expected_set:
                    mrr = 1.0 / (i + 1)
                    break

        qr = QueryResult(
            query=query_text,
            category=category,
            results=result_list,
            search_stats=stats,
            timing_ms=round(elapsed_ms, 1),
            expected_ids=expected_ids,
            precision_at_5=round(p_at_5, 3),
            recall_at_10=round(r_at_10, 3),
            mrr=round(mrr, 3),
        )
        report.query_results.append(qr)

        # Accumulate stats
        report.total_fts_hits += stats.get("fts_hits", 0)
        report.total_chroma_hits += stats.get("chroma_hits", 0)
        report.total_entity_hits += stats.get("entity_hits", 0)

    # Compute averages
    n = len(report.query_results)
    if n > 0:
        report.avg_timing_ms = round(sum(qr.timing_ms for qr in report.query_results) / n, 1)
        if ground_truth:
            scored = [qr for qr in report.query_results if qr.expected_ids]
            if scored:
                report.avg_precision_at_5 = round(sum(qr.precision_at_5 for qr in scored) / len(scored), 3)
                report.avg_recall_at_10 = round(sum(qr.recall_at_10 for qr in scored) / len(scored), 3)
                report.avg_mrr = round(sum(qr.mrr for qr in scored) / len(scored), 3)

    await sqlite.close()

    return report


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def print_discover(report: BenchmarkReport):
    """Print detailed results for ground truth creation."""
    for qr in report.query_results:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]Query: {qr.query}[/bold] ({qr.category})")
        console.print(f"[dim]Stats: chroma={qr.search_stats.get('chroma_hits', 0)} "
                       f"fts={qr.search_stats.get('fts_hits', 0)} "
                       f"entity={qr.search_stats.get('entity_hits', 0)} "
                       f"returned={qr.search_stats.get('final_returned', 0)} "
                       f"time={qr.timing_ms}ms[/dim]")

        table = Table(show_lines=False, padding=(0, 1))
        table.add_column("#", style="dim", width=3)
        table.add_column("ID", width=6)
        table.add_column("Sim", width=6)
        table.add_column("Score", width=6)
        table.add_column("Imp", width=5)
        table.add_column("Summary", min_width=50)
        table.add_column("Tags", width=20)

        for i, r in enumerate(qr.results, 1):
            tags = ", ".join(r.get("tags", [])[:3])
            table.add_row(
                str(i),
                str(r["id"]),
                f"{r['similarity']:.2f}",
                f"{r['boosted_score']:.2f}",
                f"{r['importance']:.2f}",
                r["summary"][:80],
                tags,
            )
        console.print(table)


def print_summary(report: BenchmarkReport):
    """Print summary table."""
    table = Table(title=f"Search Benchmark: {report.config_label}", show_lines=True)
    table.add_column("Query", min_width=30)
    table.add_column("Cat", width=12)
    table.add_column("Results", width=8)
    table.add_column("Chroma", width=7)
    table.add_column("FTS", width=5)
    table.add_column("Entity", width=7)
    table.add_column("Time", width=8)
    table.add_column("P@5", width=6)
    table.add_column("R@10", width=6)
    table.add_column("MRR", width=6)

    for qr in report.query_results:
        fts = qr.search_stats.get("fts_hits", 0)
        fts_style = "[red]" if fts == 0 else ""
        fts_end = "[/red]" if fts == 0 else ""

        table.add_row(
            qr.query[:35],
            qr.category,
            str(len(qr.results)),
            str(qr.search_stats.get("chroma_hits", 0)),
            f"{fts_style}{fts}{fts_end}",
            str(qr.search_stats.get("entity_hits", 0)),
            f"{qr.timing_ms:.0f}ms",
            f"{qr.precision_at_5:.2f}" if qr.expected_ids else "-",
            f"{qr.recall_at_10:.2f}" if qr.expected_ids else "-",
            f"{qr.mrr:.2f}" if qr.expected_ids else "-",
        )

    console.print(table)

    console.print(f"\n[bold]Totals:[/bold]")
    console.print(f"  Queries: {report.total_queries}")
    console.print(f"  Avg time: {report.avg_timing_ms:.0f}ms")
    console.print(f"  Total hits: chroma={report.total_chroma_hits} fts={report.total_fts_hits} entity={report.total_entity_hits}")
    if report.avg_mrr > 0:
        console.print(f"  Avg P@5: {report.avg_precision_at_5:.3f}")
        console.print(f"  Avg R@10: {report.avg_recall_at_10:.3f}")
        console.print(f"  Avg MRR: {report.avg_mrr:.3f}")


def print_comparison(report_a: BenchmarkReport, report_b: BenchmarkReport):
    """Print A/B comparison table."""
    table = Table(title="A/B Comparison", show_lines=True)
    table.add_column("Query", min_width=30)
    table.add_column(f"{report_a.config_label}", min_width=20)
    table.add_column(f"{report_b.config_label}", min_width=20)
    table.add_column("Winner", width=10)

    by_query_a = {qr.query: qr for qr in report_a.query_results}
    by_query_b = {qr.query: qr for qr in report_b.query_results}

    for query in dict.fromkeys(q.query for q in report_a.query_results):
        qa = by_query_a.get(query)
        qb = by_query_b.get(query)

        def fmt(qr):
            if not qr:
                return "-"
            fts = qr.search_stats.get("fts_hits", 0)
            return f"{len(qr.results)}r c:{qr.search_stats.get('chroma_hits',0)} f:{fts} e:{qr.search_stats.get('entity_hits',0)} {qr.timing_ms:.0f}ms"

        winner = ""
        if qa and qb and qa.expected_ids:
            if qa.mrr > qb.mrr:
                winner = report_a.config_label
            elif qb.mrr > qa.mrr:
                winner = report_b.config_label
            else:
                winner = "tie"

        table.add_row(query[:35], fmt(qa), fmt(qb), winner)

    console.print(table)

    console.print(f"\n[bold]Averages:[/bold]")
    console.print(f"  {report_a.config_label}: P@5={report_a.avg_precision_at_5:.3f} R@10={report_a.avg_recall_at_10:.3f} MRR={report_a.avg_mrr:.3f} {report_a.avg_timing_ms:.0f}ms")
    console.print(f"  {report_b.config_label}: P@5={report_b.avg_precision_at_5:.3f} R@10={report_b.avg_recall_at_10:.3f} MRR={report_b.avg_mrr:.3f} {report_b.avg_timing_ms:.0f}ms")


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def save_discover_for_ground_truth(report: BenchmarkReport, path: str):
    """Save discover results as a template for ground truth creation.

    Creates a JSON file where each query maps to its returned IDs.
    User reviews and removes incorrect IDs to create ground truth.
    """
    template = {}
    for qr in report.query_results:
        template[qr.query] = {
            "expected_ids": [r["id"] for r in qr.results[:10]],
            "results_for_review": [
                {"id": r["id"], "summary": r["summary"][:120], "score": r["boosted_score"]}
                for r in qr.results[:15]
            ],
        }

    Path(path).write_text(json.dumps(template, indent=2, default=str), encoding="utf-8")
    console.print(f"\n[bold green]Ground truth template saved to {path}[/bold green]")
    console.print("Review each query and remove incorrect IDs from expected_ids.")
    console.print("Delete the results_for_review field when done.")


def load_ground_truth(path: str) -> dict:
    """Load ground truth: {query_text: [expected_memory_ids]}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt = {}
    for query, value in data.items():
        if isinstance(value, list):
            gt[query] = value
        elif isinstance(value, dict):
            gt[query] = value.get("expected_ids", [])
    return gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Search quality benchmark — no Agent/Router/LLM required",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default="data/blipshell.db",
                        help="SQLite database path (default: data/blipshell.db)")
    parser.add_argument("--chroma", default="data/chroma",
                        help="ChromaDB directory (default: data/chroma)")
    parser.add_argument("--ollama", default="http://localhost:11434",
                        help="Ollama URL for embeddings (default: localhost:11434)")
    parser.add_argument("--output", "-o", default="data/benchmark_search_results.json",
                        help="Output JSON report path")
    parser.add_argument("--queries", default=None,
                        help="Custom queries JSON file (default: built-in queries)")
    parser.add_argument("--ground-truth", default=None,
                        help="Ground truth JSON for precision/recall scoring")
    parser.add_argument("--discover", action="store_true",
                        help="Print detailed results for ground truth creation")
    parser.add_argument("--save-template", default=None,
                        help="Save discover results as ground truth template")
    parser.add_argument("--ab", action="store_true",
                        help="A/B comparison: run baseline then overridden config")
    parser.add_argument("--project", default=None,
                        help="Scope search to a project name")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="JSON output only")

    # Search config overrides (for A/B testing)
    parser.add_argument("--similarity-threshold", type=float, default=None)
    parser.add_argument("--min-importance", type=float, default=None)
    parser.add_argument("--entity-cap", type=int, default=None,
                        help="Override MAX_ENTITY_MEMORIES in search")
    parser.add_argument("--fts-weight", type=float, default=None)
    parser.add_argument("--recency-boost", type=float, default=None)
    parser.add_argument("--dedup-threshold", type=float, default=None)

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.db):
        console.print(f"[bold red]Database not found: {args.db}[/bold red]")
        sys.exit(1)
    if not os.path.exists(args.chroma):
        console.print(f"[bold red]ChromaDB directory not found: {args.chroma}[/bold red]")
        sys.exit(1)

    # Load queries
    if args.queries:
        with open(args.queries, "r", encoding="utf-8") as f:
            queries = json.load(f)
    else:
        queries = DEFAULT_QUERIES

    # Load ground truth
    ground_truth = None
    if args.ground_truth and os.path.exists(args.ground_truth):
        ground_truth = load_ground_truth(args.ground_truth)
        if not args.quiet:
            console.print(f"[dim]Loaded ground truth: {len(ground_truth)} queries[/dim]")

    # Build config overrides
    overrides = {}
    override_label_parts = []
    if args.similarity_threshold is not None:
        overrides["similarity_threshold"] = args.similarity_threshold
        override_label_parts.append(f"sim={args.similarity_threshold}")
    if args.min_importance is not None:
        overrides["min_importance"] = args.min_importance
        override_label_parts.append(f"imp={args.min_importance}")
    if args.fts_weight is not None:
        overrides["fts_weight"] = args.fts_weight
        override_label_parts.append(f"fts={args.fts_weight}")
    if args.recency_boost is not None:
        overrides["recency_boost_weight"] = args.recency_boost
        override_label_parts.append(f"rec={args.recency_boost}")
    if args.dedup_threshold is not None:
        overrides["dedup_jaccard_threshold"] = args.dedup_threshold
        override_label_parts.append(f"dup={args.dedup_threshold}")
    override_label = "+".join(override_label_parts) if override_label_parts else "override"

    if not args.quiet:
        console.print(f"[bold]Search Quality Benchmark[/bold]")
        console.print(f"  DB: {args.db}")
        console.print(f"  ChromaDB: {args.chroma}")
        console.print(f"  Ollama: {args.ollama}")
        console.print(f"  Queries: {len(queries)}")
        console.print(f"  Project: {args.project or 'none'}")
        if overrides:
            console.print(f"  Overrides: {overrides}")
        console.print()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Run baseline
    if not args.quiet:
        console.print("[bold cyan]Running baseline...[/bold cyan]")

    report_a = await run_benchmark(
        db_path=args.db,
        chroma_path=args.chroma,
        ollama_url=args.ollama,
        queries=queries,
        ground_truth=ground_truth,
        project=args.project,
        config_label="baseline",
    )

    if args.discover:
        print_discover(report_a)
        if args.save_template:
            save_discover_for_ground_truth(report_a, args.save_template)
        elif not args.quiet:
            # Auto-save template
            template_path = args.output.replace(".json", "_template.json")
            save_discover_for_ground_truth(report_a, template_path)
    elif not args.quiet:
        print_summary(report_a)

    # A/B comparison
    report_b = None
    if args.ab and overrides:
        if not args.quiet:
            console.print(f"\n[bold cyan]Running override ({override_label})...[/bold cyan]")

        report_b = await run_benchmark(
            db_path=args.db,
            chroma_path=args.chroma,
            ollama_url=args.ollama,
            queries=queries,
            ground_truth=ground_truth,
            project=args.project,
            config_overrides=overrides,
            config_label=override_label,
        )

        if not args.quiet:
            print_summary(report_b)
            console.print()
            print_comparison(report_a, report_b)

    # Save results
    output_data = {
        "baseline": asdict(report_a),
    }
    if report_b:
        output_data["override"] = asdict(report_b)

    Path(args.output).write_text(
        json.dumps(output_data, indent=2, default=str),
        encoding="utf-8",
    )

    if not args.quiet:
        console.print(f"\n[bold green]Results saved to {args.output}[/bold green]")
    else:
        print(json.dumps(output_data, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
