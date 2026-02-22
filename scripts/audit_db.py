"""Automated health check / audit for BlipShell databases.

Checks:
1. SQLite integrity (pragma integrity_check, FK violations)
2. Memory pipeline (missing summaries, score distributions, unscored)
3. Entity quality (LLM artifacts, orphans, invalid types)
4. ChromaDB/SQLite sync (count mismatches, missing embeddings)
5. Session health (message count accuracy, orphaned memories)
6. Tag coverage (orphaned tags, untagged memories)
7. FTS5 sync (row count match)

Usage:
    python scripts/audit_db.py
    python scripts/audit_db.py --db data/blipshell.db --chroma data/chroma
    python scripts/audit_db.py --json           # machine-readable output
    python scripts/audit_db.py --fix            # auto-fix safe issues (future)
"""

import argparse
import json
import re
import sqlite3
import sys
import urllib.request
import urllib.error
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# --- Entity quality patterns (subset from cleanup_entities.py) ---

VALID_ENTITY_TYPES = {
    "person", "project", "technology", "concept",
    "preference", "place", "organization",
}

PRONOUN_NAMES = {
    "she", "her", "he", "him", "his", "they", "them", "their",
    "it", "its", "this", "that", "these", "those",
    "something", "someone", "anything", "nothing",
    "the user", "the assistant", "assistant", "none",
    "subject", "object", "predicate",
}

COMMENTARY_PATTERNS = [
    re.compile(r"</think>", re.IGNORECASE),
    re.compile(r"<think>", re.IGNORECASE),
    re.compile(r"\bi think\b", re.IGNORECASE),
    re.compile(r"\blet me\b", re.IGNORECASE),
    re.compile(r"\bhere are\b", re.IGNORECASE),
    re.compile(r"\bfinal answer", re.IGNORECASE),
    re.compile(r"\bfinal output", re.IGNORECASE),
    re.compile(r"\bthe memory\b", re.IGNORECASE),
    re.compile(r"\bcould be[:\s]", re.IGNORECASE),
    re.compile(r"\bmaybe[:\s]", re.IGNORECASE),
    re.compile(r"\btriple[s]?\b", re.IGNORECASE),
    re.compile(r"\bextract\b", re.IGNORECASE),
    re.compile(r"\u2192"),  # → arrow
    re.compile(r"->"),
]


def severity_color(sev: str) -> str:
    return {"ok": "green", "info": "blue", "warn": "yellow", "error": "red"}[sev]


class AuditResult:
    """Collects findings from all checks."""

    def __init__(self):
        self.findings: list[dict] = []

    def add(self, category: str, check: str, severity: str, message: str, detail: str = ""):
        self.findings.append({
            "category": category,
            "check": check,
            "severity": severity,
            "message": message,
            "detail": detail,
        })

    def print_report(self):
        table = Table(title="BlipShell Database Audit", show_lines=True)
        table.add_column("Category", style="bold")
        table.add_column("Check")
        table.add_column("Severity")
        table.add_column("Message")

        for f in self.findings:
            sev = f["severity"]
            table.add_row(
                f["category"],
                f["check"],
                f"[{severity_color(sev)}]{sev.upper()}[/{severity_color(sev)}]",
                f["message"],
            )

        console.print(table)

        # Summary
        counts = {}
        for f in self.findings:
            counts[f["severity"]] = counts.get(f["severity"], 0) + 1
        parts = []
        for sev in ["error", "warn", "info", "ok"]:
            if sev in counts:
                parts.append(f"[{severity_color(sev)}]{counts[sev]} {sev}[/{severity_color(sev)}]")
        console.print(f"\nSummary: {', '.join(parts)}")

    def to_json(self) -> str:
        return json.dumps(self.findings, indent=2)


def check_sqlite_integrity(db_path: str, result: AuditResult):
    """Run pragma integrity_check and foreign_key_check."""
    conn = sqlite3.connect(db_path)
    try:
        # Integrity check
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        if rows == [("ok",)]:
            result.add("SQLite", "integrity_check", "ok", "Database integrity OK")
        else:
            msgs = [r[0] for r in rows[:5]]
            result.add("SQLite", "integrity_check", "error",
                       f"Integrity issues found: {len(rows)}", "; ".join(msgs))

        # Foreign key check
        conn.execute("PRAGMA foreign_keys = ON")
        fk_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
        if not fk_rows:
            result.add("SQLite", "foreign_key_check", "ok", "No FK violations")
        else:
            tables = set(r[0] for r in fk_rows)
            result.add("SQLite", "foreign_key_check", "warn",
                       f"{len(fk_rows)} FK violations in: {', '.join(tables)}")
    finally:
        conn.close()


def check_memory_pipeline(db_path: str, result: AuditResult):
    """Check memory summaries, scores, and distributions."""
    conn = sqlite3.connect(db_path)
    try:
        # Total memories (non-archived)
        total = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE is_archived = 0"
        ).fetchone()[0]
        total_all = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        archived = total_all - total
        result.add("Memory", "counts", "info",
                    f"{total} active, {archived} archived, {total_all} total")

        # Missing summaries
        no_summary = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE summary IS NULL AND is_archived = 0"
        ).fetchone()[0]
        if no_summary == 0:
            result.add("Memory", "summaries", "ok", "All active memories have summaries")
        elif no_summary < total * 0.01:
            result.add("Memory", "summaries", "info",
                       f"{no_summary} active memories missing summaries ({no_summary/total*100:.1f}%)")
        else:
            result.add("Memory", "summaries", "warn",
                       f"{no_summary} active memories missing summaries ({no_summary/total*100:.1f}%)")

        # Unscored memories (rank=0 or importance=0.0)
        unranked = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE rank = 0 AND is_archived = 0"
        ).fetchone()[0]
        unimportant = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE importance = 0.0 AND is_archived = 0"
        ).fetchone()[0]
        if unranked == 0:
            result.add("Memory", "rank_coverage", "ok", "All active memories are ranked")
        else:
            pct = unranked / total * 100 if total else 0
            sev = "warn" if pct > 5 else "info"
            result.add("Memory", "rank_coverage", sev,
                       f"{unranked} memories with rank=0 ({pct:.1f}%)")

        if unimportant == 0:
            result.add("Memory", "importance_coverage", "ok",
                       "All active memories have importance > 0")
        else:
            pct = unimportant / total * 100 if total else 0
            sev = "warn" if pct > 5 else "info"
            result.add("Memory", "importance_coverage", sev,
                       f"{unimportant} memories with importance=0.0 ({pct:.1f}%)")

        # Rank distribution
        rank_dist = conn.execute("""
            SELECT rank, COUNT(*) FROM memories
            WHERE is_archived = 0 AND rank > 0
            GROUP BY rank ORDER BY rank
        """).fetchall()
        if rank_dist:
            dist_str = ", ".join(f"rank {r}: {c}" for r, c in rank_dist)
            result.add("Memory", "rank_distribution", "info", dist_str)

        # Importance distribution (quartiles)
        imp_stats = conn.execute("""
            SELECT
                MIN(importance), AVG(importance), MAX(importance),
                COUNT(CASE WHEN importance < 0.3 THEN 1 END),
                COUNT(CASE WHEN importance >= 0.3 AND importance < 0.6 THEN 1 END),
                COUNT(CASE WHEN importance >= 0.6 THEN 1 END)
            FROM memories WHERE is_archived = 0 AND importance > 0.0
        """).fetchone()
        if imp_stats and imp_stats[0] is not None:
            result.add("Memory", "importance_distribution", "info",
                       f"min={imp_stats[0]:.2f} avg={imp_stats[1]:.2f} max={imp_stats[2]:.2f} "
                       f"| low(<0.3)={imp_stats[3]} mid(0.3-0.6)={imp_stats[4]} high(>0.6)={imp_stats[5]}")

        # Core memories and lessons
        core_count = conn.execute(
            "SELECT COUNT(*) FROM core_memories WHERE is_active = 1"
        ).fetchone()[0]
        lesson_count = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        result.add("Memory", "core_and_lessons", "info",
                    f"{core_count} active core memories, {lesson_count} lessons")

    finally:
        conn.close()


def check_entity_quality(db_path: str, result: AuditResult):
    """Check entity names for LLM artifacts and invalid types."""
    conn = sqlite3.connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        result.add("Entities", "total_count", "info", f"{total} entities")

        # Check for commentary patterns
        entities = conn.execute("SELECT id, name, entity_type FROM entities").fetchall()
        commentary_count = 0
        pronoun_count = 0
        invalid_type_count = 0
        single_char = 0
        long_names = 0
        think_tags = 0

        for eid, name, etype in entities:
            name_lower = name.lower().strip()
            if name_lower in PRONOUN_NAMES:
                pronoun_count += 1
            if any(p.search(name) for p in COMMENTARY_PATTERNS):
                commentary_count += 1
            if "</think>" in name.lower() or "<think>" in name.lower():
                think_tags += 1
            if etype and etype.lower() not in VALID_ENTITY_TYPES:
                invalid_type_count += 1
            if len(name.strip()) <= 1:
                single_char += 1
            if len(name) > 60:
                long_names += 1

        issues = []
        if think_tags:
            issues.append(f"{think_tags} with <think> tags")
        if commentary_count:
            issues.append(f"{commentary_count} with LLM commentary")
        if pronoun_count:
            issues.append(f"{pronoun_count} pronoun/vague names")
        if invalid_type_count:
            issues.append(f"{invalid_type_count} invalid entity types")
        if single_char:
            issues.append(f"{single_char} single-char names")
        if long_names:
            issues.append(f"{long_names} names > 60 chars")

        if not issues:
            result.add("Entities", "quality", "ok", "No LLM artifact issues found")
        else:
            sev = "error" if think_tags else "warn"
            result.add("Entities", "quality", sev, "; ".join(issues))

        # Orphaned entities (no mentions AND no relationships)
        orphaned = conn.execute("""
            SELECT COUNT(*) FROM entities e
            WHERE NOT EXISTS (SELECT 1 FROM entity_mentions em WHERE em.entity_id = e.id)
            AND NOT EXISTS (SELECT 1 FROM entity_relationships er
                           WHERE er.subject_id = e.id OR er.object_id = e.id)
        """).fetchone()[0]
        if orphaned == 0:
            result.add("Entities", "orphans", "ok", "No orphaned entities")
        else:
            pct = orphaned / total * 100 if total else 0
            sev = "warn" if pct > 5 else "info"
            result.add("Entities", "orphans", sev,
                       f"{orphaned} orphaned entities ({pct:.1f}%)")

        # Relationship and mention counts
        rel_count = conn.execute("SELECT COUNT(*) FROM entity_relationships").fetchone()[0]
        mention_count = conn.execute("SELECT COUNT(*) FROM entity_mentions").fetchone()[0]
        result.add("Entities", "graph_size", "info",
                    f"{rel_count} relationships, {mention_count} mentions")

    finally:
        conn.close()


def check_chroma_sync(db_path: str, chroma_path: str, result: AuditResult):
    """Check ChromaDB collection counts vs SQLite."""
    if not Path(chroma_path).exists():
        result.add("ChromaDB", "exists", "warn", f"ChromaDB directory not found: {chroma_path}")
        return

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

        conn = sqlite3.connect(db_path)
        try:
            # Memories collection
            try:
                memories_col = client.get_collection("memories")
                chroma_mem_count = memories_col.count()
            except Exception:
                chroma_mem_count = 0
                result.add("ChromaDB", "memories_collection", "error",
                           "Cannot access memories collection")

            # SQLite memories with summaries (should be embedded)
            sqlite_mem_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE summary IS NOT NULL AND is_archived = 0"
            ).fetchone()[0]

            if chroma_mem_count > 0:
                diff = abs(chroma_mem_count - sqlite_mem_count)
                pct = diff / sqlite_mem_count * 100 if sqlite_mem_count else 0
                if pct < 1:
                    result.add("ChromaDB", "memories_sync", "ok",
                               f"Memories: ChromaDB={chroma_mem_count}, SQLite={sqlite_mem_count}")
                elif pct < 5:
                    result.add("ChromaDB", "memories_sync", "info",
                               f"Memories: ChromaDB={chroma_mem_count}, SQLite={sqlite_mem_count} (diff={diff})")
                else:
                    result.add("ChromaDB", "memories_sync", "warn",
                               f"Memories out of sync: ChromaDB={chroma_mem_count}, SQLite={sqlite_mem_count} (diff={diff})")

            # Core memories collection
            try:
                core_col = client.get_collection("core_memories")
                chroma_core = core_col.count()
            except Exception:
                chroma_core = 0

            sqlite_core = conn.execute(
                "SELECT COUNT(*) FROM core_memories WHERE is_active = 1"
            ).fetchone()[0]
            result.add("ChromaDB", "core_sync", "info",
                       f"Core: ChromaDB={chroma_core}, SQLite={sqlite_core}")

            # Lessons collection
            try:
                lessons_col = client.get_collection("lessons")
                chroma_lessons = lessons_col.count()
            except Exception:
                chroma_lessons = 0

            sqlite_lessons = conn.execute(
                "SELECT COUNT(*) FROM lessons"
            ).fetchone()[0]
            result.add("ChromaDB", "lessons_sync", "info",
                       f"Lessons: ChromaDB={chroma_lessons}, SQLite={sqlite_lessons}")

        finally:
            conn.close()

    except ImportError:
        result.add("ChromaDB", "import", "warn", "chromadb not installed, skipping sync check")
    except Exception as e:
        result.add("ChromaDB", "error", "error", f"ChromaDB check failed: {e}")


def check_sessions(db_path: str, result: AuditResult):
    """Check session health: message counts, orphaned memories."""
    conn = sqlite3.connect(db_path)
    try:
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        archived = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE is_archived = 1"
        ).fetchone()[0]
        result.add("Sessions", "counts", "info",
                    f"{total_sessions} sessions ({archived} archived)")

        # Sessions with wrong message_count
        mismatches = conn.execute("""
            SELECT s.id, s.message_count,
                   (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.id) as actual
            FROM sessions s
            WHERE s.message_count != (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.id)
        """).fetchall()
        if not mismatches:
            result.add("Sessions", "message_counts", "ok",
                       "All session message counts match")
        else:
            result.add("Sessions", "message_counts", "info",
                       f"{len(mismatches)} sessions with mismatched message_count")

        # Orphaned memories (session_id doesn't exist)
        orphaned = conn.execute("""
            SELECT COUNT(*) FROM memories m
            WHERE m.session_id IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.id = m.session_id)
        """).fetchone()[0]
        if orphaned == 0:
            result.add("Sessions", "orphaned_memories", "ok", "No orphaned memories")
        else:
            result.add("Sessions", "orphaned_memories", "warn",
                       f"{orphaned} memories reference non-existent sessions")

        # Empty sessions (no memories at all)
        empty = conn.execute("""
            SELECT COUNT(*) FROM sessions s
            WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.session_id = s.id)
        """).fetchone()[0]
        if empty > 0:
            result.add("Sessions", "empty_sessions", "info",
                       f"{empty} sessions with no memories")

    finally:
        conn.close()


def check_tags(db_path: str, result: AuditResult):
    """Check tag coverage and orphaned tags."""
    conn = sqlite3.connect(db_path)
    try:
        total_tags = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        result.add("Tags", "count", "info", f"{total_tags} tags")

        if total_tags == 0:
            result.add("Tags", "coverage", "info", "No tags exist yet")
            return

        # Orphaned tags (not linked to any memory, core memory, or lesson)
        orphaned = conn.execute("""
            SELECT COUNT(*) FROM tags t
            WHERE NOT EXISTS (SELECT 1 FROM memory_tags mt WHERE mt.tag_id = t.id)
            AND NOT EXISTS (SELECT 1 FROM core_memory_tags ct WHERE ct.tag_id = t.id)
            AND NOT EXISTS (SELECT 1 FROM lesson_tags lt WHERE lt.tag_id = t.id)
        """).fetchone()[0]
        if orphaned == 0:
            result.add("Tags", "orphaned", "ok", "No orphaned tags")
        else:
            pct = orphaned / total_tags * 100 if total_tags else 0
            result.add("Tags", "orphaned", "info",
                       f"{orphaned} orphaned tags ({pct:.1f}%)")

        # Tag coverage on active memories
        total_active = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE is_archived = 0"
        ).fetchone()[0]
        tagged = conn.execute("""
            SELECT COUNT(DISTINCT mt.memory_id) FROM memory_tags mt
            JOIN memories m ON m.id = mt.memory_id WHERE m.is_archived = 0
        """).fetchone()[0]
        if total_active > 0:
            pct = tagged / total_active * 100
            sev = "ok" if pct > 80 else ("info" if pct > 50 else "warn")
            result.add("Tags", "memory_coverage", sev,
                       f"{tagged}/{total_active} active memories tagged ({pct:.1f}%)")

        # Case-insensitive duplicates
        dupes = conn.execute("""
            SELECT LOWER(name), category, COUNT(*) as cnt
            FROM tags GROUP BY LOWER(name), category
            HAVING cnt > 1
        """).fetchall()
        if dupes:
            result.add("Tags", "duplicates", "info",
                       f"{len(dupes)} case-insensitive duplicate tag groups")

    finally:
        conn.close()


def check_fts_sync(db_path: str, result: AuditResult):
    """Check FTS5 index is in sync with memories table."""
    conn = sqlite3.connect(db_path)
    try:
        fts_count = conn.execute(
            "SELECT COUNT(*) FROM memories_fts"
        ).fetchone()[0]
        sqlite_count = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE summary IS NOT NULL"
        ).fetchone()[0]

        diff = abs(fts_count - sqlite_count)
        if diff == 0:
            result.add("FTS5", "sync", "ok",
                       f"FTS5 in sync ({fts_count} rows)")
        elif diff < sqlite_count * 0.01:
            result.add("FTS5", "sync", "info",
                       f"FTS5={fts_count}, SQLite summaries={sqlite_count} (diff={diff})")
        else:
            result.add("FTS5", "sync", "warn",
                       f"FTS5 out of sync: FTS5={fts_count}, SQLite summaries={sqlite_count}")
    except Exception as e:
        result.add("FTS5", "sync", "error", f"FTS5 check failed: {e}")
    finally:
        conn.close()


def check_db_size(db_path: str, chroma_path: str, result: AuditResult):
    """Report database file sizes."""
    db = Path(db_path)
    if db.exists():
        size_mb = db.stat().st_size / (1024 * 1024)
        result.add("Storage", "sqlite_size", "info", f"SQLite: {size_mb:.1f} MB")

    chroma = Path(chroma_path)
    if chroma.exists():
        total = sum(f.stat().st_size for f in chroma.rglob("*") if f.is_file())
        size_mb = total / (1024 * 1024)
        result.add("Storage", "chroma_size", "info", f"ChromaDB: {size_mb:.1f} MB")


def check_endpoint_health(result: AuditResult, config_path: str | None = None):
    """Check endpoint reachability and model availability.

    Loads config.yaml to discover endpoints, then pings each one.
    For Ollama endpoints, verifies configured models are available.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from blipshell.core.config import ConfigManager
        config_mgr = ConfigManager(config_path)
        config = config_mgr.load()
    except Exception as e:
        result.add("Endpoints", "config_load", "warn", f"Cannot load config: {e}")
        return

    endpoints = config.endpoints
    models_config = config.models

    if not endpoints:
        result.add("Endpoints", "count", "info", "No endpoints configured")
        return

    result.add("Endpoints", "count", "info",
               f"{len(endpoints)} endpoint(s) configured")

    for ep_cfg in endpoints:
        name = ep_cfg.name
        url = ep_cfg.url.rstrip("/")
        provider = ep_cfg.provider
        enabled = ep_cfg.enabled

        if not enabled:
            result.add("Endpoints", f"{name}_status", "info",
                       f"{name} ({url}): disabled in config")
            continue

        # Ping the endpoint
        try:
            if provider == "ollama":
                # Ollama: GET /api/tags returns list of models
                req = urllib.request.Request(f"{url}/api/tags", method="GET")
                req.add_header("Accept", "application/json")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    available_models = {
                        m.get("name", "").split(":")[0]
                        for m in data.get("models", [])
                    }
                    # Also keep full names (with tags)
                    available_full = {m.get("name", "") for m in data.get("models", [])}

                result.add("Endpoints", f"{name}_status", "ok",
                           f"{name} ({url}): up, {len(available_full)} models")

                # Check that configured models are available
                # Collect models this endpoint should serve
                expected_models = set()
                for role in ep_cfg.roles:
                    if ep_cfg.models and role in ep_cfg.models:
                        expected_models.add(ep_cfg.models[role])
                    else:
                        # Fall back to global model config
                        model_name = getattr(models_config, role, None)
                        if model_name:
                            expected_models.add(model_name)

                missing = []
                for model in sorted(expected_models):
                    base = model.split(":")[0]
                    if model not in available_full and base not in available_models:
                        missing.append(model)

                if missing:
                    result.add("Endpoints", f"{name}_models", "warn",
                               f"{name}: missing models: {', '.join(missing)}")
                elif expected_models:
                    result.add("Endpoints", f"{name}_models", "ok",
                               f"{name}: all {len(expected_models)} expected model(s) found")

            elif provider == "openai":
                # OpenAI-compatible: just check that the base URL responds
                # Try /v1/models or just a GET to the base URL
                try:
                    req = urllib.request.Request(f"{url}/v1/models", method="GET")
                    api_key = ep_cfg.api_key
                    if api_key:
                        # Resolve ${ENV_VAR} syntax
                        from blipshell.models.config import resolve_env_vars
                        api_key = resolve_env_vars(api_key)
                        req.add_header("Authorization", f"Bearer {api_key}")
                    with urllib.request.urlopen(req, timeout=5):
                        pass
                    result.add("Endpoints", f"{name}_status", "ok",
                               f"{name} ({url}): up (openai-compatible)")
                except urllib.error.HTTPError as e:
                    if e.code in (401, 403):
                        result.add("Endpoints", f"{name}_status", "ok",
                                   f"{name} ({url}): reachable (auth may need check)")
                    else:
                        result.add("Endpoints", f"{name}_status", "warn",
                                   f"{name} ({url}): HTTP {e.code}")
            else:
                result.add("Endpoints", f"{name}_status", "info",
                           f"{name}: unknown provider '{provider}'")

        except urllib.error.URLError as e:
            result.add("Endpoints", f"{name}_status", "error",
                       f"{name} ({url}): unreachable — {e.reason}")
        except Exception as e:
            result.add("Endpoints", f"{name}_status", "error",
                       f"{name} ({url}): check failed — {e}")


def run_audit(
    db_path: str = "data/blipshell.db",
    chroma_path: str = "data/chroma",
    config_path: str | None = None,
    skip_chroma: bool = False,
    skip_endpoints: bool = False,
) -> AuditResult:
    """Run the full audit programmatically and return the result.

    Args:
        db_path: Path to SQLite database.
        chroma_path: Path to ChromaDB directory.
        config_path: Path to config.yaml (for endpoint checks).
        skip_chroma: Skip the slow ChromaDB sync check.
        skip_endpoints: Skip endpoint health checks.

    Returns:
        AuditResult with all findings.
    """
    result = AuditResult()

    if not Path(db_path).exists():
        result.add("SQLite", "exists", "error", f"Database not found: {db_path}")
        return result

    check_db_size(db_path, chroma_path, result)
    check_sqlite_integrity(db_path, result)
    check_memory_pipeline(db_path, result)
    check_entity_quality(db_path, result)
    check_sessions(db_path, result)
    check_tags(db_path, result)
    check_fts_sync(db_path, result)

    if not skip_chroma:
        check_chroma_sync(db_path, chroma_path, result)

    if not skip_endpoints:
        check_endpoint_health(result, config_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="BlipShell database health check")
    parser.add_argument("--db", default="data/blipshell.db", help="SQLite DB path")
    parser.add_argument("--chroma", default="data/chroma", help="ChromaDB directory")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--skip-chroma", action="store_true", help="Skip ChromaDB sync check")
    parser.add_argument("--skip-endpoints", action="store_true", help="Skip endpoint health checks")
    args = parser.parse_args()

    if not Path(args.db).exists():
        console.print(f"[red]Database not found: {args.db}[/red]")
        sys.exit(1)

    console.print("[bold]BlipShell Database Audit[/bold]\n")

    result = run_audit(
        db_path=args.db,
        chroma_path=args.chroma,
        config_path=args.config,
        skip_chroma=args.skip_chroma,
        skip_endpoints=args.skip_endpoints,
    )

    if args.json:
        print(result.to_json())
    else:
        result.print_report()

    # Exit code: 1 if any errors, 0 otherwise
    has_errors = any(f["severity"] == "error" for f in result.findings)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
