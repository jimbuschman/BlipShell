"""Continuity set runner — does the right evidence reach the request? (V3 Stage C)

Measures the ingest-to-request path, not the model. For each case in
tests/benchmark_continuity.py it bootstraps a REAL agent against a throwaway
database (real SQLiteStore, real VectorStore with a deterministic embedder,
real MemorySearch, real pools, real _build_messages), plants the case's
memories, asks the question, and inspects the request the chat client was
handed. Two rates come out:

- **survival**: the answer-bearing text is in the request (retrieved, packed,
  serialised, sent — the review's four stages, collapsed to the one that
  matters);
- **exclusion**: superseded / other-project / other-person / speculative text
  is absent, or present but LABELLED on the same rendered line (superseded,
  speaker, project, age). Unlabelled presence is false recall. The criterion
  is appropriate-to-the-question, so a history question expects the
  superseded fact to appear.

No model is called: the chat client answers "ok" and every router.generate()
is canned. That is the point — this measures CONTEXT DELIVERY (what reached
the request), on the dev box, in seconds, and is the Stage B / E1 gate.
It says nothing about what a model DOES with the request: behavioural
results come from real-model runs (simulate scenarios, the Tailscale model
half) and are reported separately, never merged into these numbers.

Seeds may be planted directly or driven through the production write path
(`Seed.via="pipeline"`): processor.process_message with the dedup verdict
scripted per seed, so a supersession record is created by production code -
never by fixture metadata. The headless config lowers the dedup similarity
threshold because the deterministic embedder is not semantic; the verdict
itself is the only scripted part.

Run: `python -m blipshell.benchmark.continuity` prints the table and writes
benchmark_results/continuity__<sha>__<ts>.json. The pytest wrapper is
tests/test_continuity_set.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock

from blipshell.benchmark.harness import _REPO_ROOT, _load_dataset

ALL_ROLES = [
    "tool_calling", "coding", "reasoning", "summarization", "ranking",
    "importance", "ranking_importance", "session_review", "reflection", "embedding",
]


class RecordingChatClient:
    """The chat endpoint for a headless agent: records every request, replies `reply`."""

    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.sent: list[list[dict]] = []

    def _chunk(self, messages):
        self.sent.append([dict(m) for m in messages])
        return {"message": {"content": self.reply, "tool_calls": None}, "done": True}

    async def chat_stream(self, messages, model, tools=None, **kwargs):
        yield self._chunk(messages)

    async def chat(self, messages, model, tools=None, **kwargs):
        return self._chunk(messages)


def canned_generate(task_type, prompt: str = "", system: Optional[str] = None, **kwargs) -> str:
    """Background LLM calls the agent may make during a turn, answered inertly."""
    sys_l = (system or "").lower()
    tt = str(getattr(task_type, "value", task_type))
    if sys_l.startswith("you decide what to do with a new memory"):
        return "ADD"
    if "contradict" in sys_l:
        return "NO"
    if tt == "ranking_importance":
        return "3 0.5 conversation"
    if tt == "summarization":
        # echo the message so the stored summary is the text, not the prompt
        body = (prompt or "").split("Summarize this message:", 1)[-1].strip()
        return body[:200] if body else "SKIP"
    return "NO"  # relevance judges (self-thought resurfacing) stay closed


async def bootstrap_headless_agent(db_path: str | Path, *, reply: str = "ok",
                                   config=None):
    """A real Agent with real stores and search, no network, no background tasks.

    Returns (agent, client). Uses Agent._build_subsystems (the DB-only half of
    initialize) and then swaps: the vector store's embedder for the
    deterministic fake in tests/fakes.py, every endpoint's client for a
    RecordingChatClient, and router.generate for canned_generate.
    """
    from blipshell.core.agent import Agent
    from blipshell.core.config import ConfigManager
    from blipshell.models.config import BlipShellConfig, EndpointConfig

    cfg = config or BlipShellConfig()
    cfg.database.path = str(db_path)
    cfg.database.require_existing = False
    cfg.endpoints = [EndpointConfig(
        name="scripted", url="http://127.0.0.1:9", provider="ollama",
        roles=list(ALL_ROLES), priority=1, max_concurrent=4, context_tokens=32768,
    )]
    cfg.reflection.enabled = False
    cfg.robotics.enabled = False
    # The deterministic embedder is not semantic (a correction and the fact
    # it corrects score ~0.18), so with the production 0.7 candidate bar the
    # dedup step would never ask for a verdict. In the harness the nearest
    # memories ARE the candidates: threshold 0 here only. The verdict is
    # scripted per seed; everything after it is production code, including
    # the scope check that protects other projects.
    cfg.memory.dedup.similarity_threshold = 0.0

    agent = Agent(cfg, ConfigManager(None))
    await agent._build_subsystems()

    fakes = _load_dataset("fakes")
    fakes.install_fake_embedder(agent.vectors, cfg.database.embedding_dimensions)

    client = RecordingChatClient(reply)
    for ep in agent.endpoint_manager._endpoints:
        ep.client = client
    agent.router.generate = AsyncMock(side_effect=canned_generate)
    agent._initialized = True
    return agent, client


@dataclass
class CaseResult:
    name: str
    family: str
    passed: bool
    missing: list[str] = field(default_factory=list)          # must_appear not found
    unlabelled: list[str] = field(default_factory=list)       # forbidden text present without its label
    labelled: list[str] = field(default_factory=list)         # forbidden text present WITH its label (ok)
    leaked: list[str] = field(default_factory=list)           # must_not_appear found
    request_chars: int = 0
    # How many times a planted memory's text was rendered beyond once. The
    # baseline showed every recalled memory twice - once in Recall with a time
    # label, once in RecentHistory without - which is the pool-duplication
    # finding Stage B1 is about; this number is its gauge.
    duplicated_renders: int = 0


def _verdict_router(verdict: str):
    """router.generate side-effect: the dedup verdict is `verdict`, all else canned."""
    async def gen(task_type, prompt="", system=None, **kwargs):
        if (system or "").lower().startswith("you decide what to do with a new memory"):
            return verdict
        return canned_generate(task_type, prompt=prompt, system=system, **kwargs)
    return gen


async def seed_case(agent, case, now: Optional[datetime] = None) -> None:
    from blipshell.models.memory import CoreMemory, Lesson, Memory, MemoryType

    now = now or datetime.now(timezone.utc)
    sessions: dict[str, int] = {}
    decision_ids: dict[str, int] = {}
    for seed in case.seeds:
        ts = now - timedelta(days=seed.days_ago)
        if seed.session not in sessions:
            sessions[seed.session] = await agent.sqlite.create_session(
                title=f"seed {seed.session}", project=seed.project, created_at=ts,
            )
        sid = sessions[seed.session]
        if seed.kind == "memory" and getattr(seed, "via", "direct") == "pipeline":
            # The REAL write path, with only the model's verdict scripted.
            agent.router.generate = AsyncMock(side_effect=_verdict_router(getattr(seed, "dedup_verdict", "ADD")))
            try:
                mid = await agent.processor.process_message(
                    seed.content, role=seed.role, session_id=sid, timestamp=ts,
                )
            finally:
                agent.router.generate = AsyncMock(side_effect=canned_generate)
            if mid is None:
                raise RuntimeError(f"pipeline seed was filtered as noise: {seed.content[:60]!r}")
            # the pipeline stamps 'now'; the case wants the seed's age
            await agent.sqlite._db.execute("UPDATE memories SET timestamp = ? WHERE id = ?",
                                           (ts.isoformat(), mid))
            await agent.sqlite._db.commit()
        elif seed.kind == "memory":
            mid = await agent.sqlite.create_memory(Memory(
                session_id=sid, role=seed.role, content=seed.content,
                summary=seed.content[:200], timestamp=ts, rank=3, importance=0.6,
                memory_type=MemoryType.CONVERSATION,
            ))
            agent.vectors.add_memory(mid, seed.content, {"session_id": str(sid), "role": seed.role})
        elif seed.kind == "decision":
            from blipshell.memory import decisions
            label = getattr(seed, "label", "") or seed.session
            target = getattr(seed, "supersedes_seed", None)
            if target:
                new = await decisions.revise_decision(
                    agent.sqlite, agent.vectors, decision_ids[target], decision=seed.content,
                    reason=getattr(seed, "reason", ""), revisit_when=getattr(seed, "revisit_when", ""),
                    session_id=sid, decided_by=seed.role,
                )
            else:
                new = await decisions.record_decision(
                    agent.sqlite, agent.vectors, decision=seed.content,
                    reason=getattr(seed, "reason", ""), revisit_when=getattr(seed, "revisit_when", ""),
                    project=seed.project, session_id=sid, decided_by=seed.role,
                )
            decision_ids[label] = new.id
            await agent.sqlite._db.execute("UPDATE memories SET timestamp = ? WHERE id = ?",
                                           (ts.isoformat(), new.id))
            await agent.sqlite._db.commit()
        elif seed.kind == "core":
            cid = await agent.sqlite.create_core_memory(CoreMemory(content=seed.content, importance=0.8))
            agent.vectors.add_core_memory(cid, seed.content)
        elif seed.kind == "lesson":
            lid = await agent.sqlite.create_lesson(Lesson(content=seed.content, importance=0.6))
            agent.vectors.add_lesson(lid, seed.content)
        elif seed.kind == "followup":
            fid = await agent.sqlite.add_follow_up(seed.content, session_id=sid, project=seed.project,
                                                   due_hint=getattr(seed, "reason", "") or None)
            from blipshell.memory import project_events
            await project_events.record_event(agent.sqlite, project=seed.project, kind="followup_added",
                                              summary=seed.content, ref_kind="follow_up", ref_id=fid, session_id=sid)
        elif seed.kind == "task_event":
            from blipshell.memory import project_events
            await project_events.record_event(agent.sqlite, project=seed.project, kind="task_completed",
                                              summary=seed.content, session_id=sid,
                                              source_type="assistant_inference")
        else:
            raise ValueError(f"unknown seed kind {seed.kind!r}")
    if case.active_project:
        # A real project row with a real (empty) root, and the project context
        # the activation path builds - so the dossier reaches the request the
        # way it does in production (E2).
        root = Path(tempfile.mkdtemp(prefix="blipshell_proj_"))
        if not await agent.sqlite.get_project(case.active_project):
            await agent.sqlite.create_project(case.active_project, root_path=str(root))
        else:
            await agent.sqlite.update_project(case.active_project, root_path=str(root))
        project_row = await agent.sqlite.get_project(case.active_project)
        agent.active_project = {"name": case.active_project, "root_path": str(root)}
        agent._project_context = (await agent._scan_project_context(project_row)
                                  + await agent._dossier_context(project_row))


def _request_text(messages: list[dict], question: Optional[str] = None) -> str:
    """Everything the model was sent EXCEPT the current user turn: the question
    itself legitimately contains its own words ("Do I prefer tabs or spaces?")
    and must not count as recalled evidence."""
    parts = []
    for i, m in enumerate(messages):
        content = str(m.get("content") or "")
        if question and i == len(messages) - 1 and m.get("role") == "user" and content.endswith(question):
            continue
        parts.append(content)
    return "\n".join(parts)


def _has_label(line: str, label: str) -> bool:
    if label.startswith("re:"):
        import re
        return re.search(label[3:], line, flags=re.IGNORECASE) is not None
    return label.lower() in line.lower()


def score_request(case, messages: list[dict]) -> CaseResult:
    text = _request_text(messages, getattr(case, "question", None))
    lines = text.splitlines()
    missing = [s for s in case.must_appear if s not in text]
    labelled, unlabelled = [], []
    for sub, label in case.forbidden_unless_labelled:
        if sub in text:
            ok = any(sub in ln and _has_label(ln, label) for ln in lines)
            (labelled if ok else unlabelled).append(sub)
    leaked = [s for s in case.must_not_appear if s in text]
    dup = 0
    for seed in case.seeds:
        # The TAIL of the content: the buried-fact seed is 1300 chars of
        # repeated filler followed by the fact, so a head probe counts the
        # filler's own repetitions.
        probe = seed.content[-60:]
        dup += max(0, text.count(probe) - 1)
    return CaseResult(
        name=case.name, family=case.family,
        passed=not missing and not unlabelled and not leaked,
        missing=missing, unlabelled=unlabelled, labelled=labelled, leaked=leaked,
        request_chars=len(text), duplicated_renders=dup,
    )


async def run_case(case, *, keep_db: bool = False) -> CaseResult:
    tmp = tempfile.mkdtemp(prefix="blipshell_continuity_")
    db = Path(tmp) / "case.db"
    agent, client = await bootstrap_headless_agent(db)
    try:
        await seed_case(agent, case)
        await agent.start_session()
        client.sent.clear()
        await agent.chat(case.question)
        sent = client.sent[0] if client.sent else []
        return score_request(case, sent)
    finally:
        # Raw-memory persists run as tracked tasks; closing under them logs
        # "Cannot operate on a closed database" for every turn.
        try:
            await agent.session_manager.flush_pending_persists()
        except Exception:
            pass
        try:
            await agent.sqlite.close()
        except Exception:
            pass
        try:
            agent.vectors.close()
        except Exception:
            pass
        if not keep_db:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


def summarize(results: list[CaseResult]) -> dict:
    surv = [r for r in results if r.family == "survival"]
    fr = [r for r in results if r.family == "false_recall"]
    n_lab = sum(len(r.labelled) for r in fr)
    n_unlab = sum(len(r.unlabelled) for r in fr)
    return {
        "cases": len(results),
        "survival_rate": round(sum(r.passed for r in surv) / len(surv), 4) if surv else None,
        "exclusion_rate": round(sum(r.passed for r in fr) / len(fr), 4) if fr else None,
        # informational until B4 labels exist: share of forbidden text that
        # surfaced WITH its label rather than without
        "labelled_rate": round(n_lab / (n_lab + n_unlab), 4) if (n_lab + n_unlab) else None,
        "duplicated_renders": sum(r.duplicated_renders for r in results),
        "failed": [r.name for r in results if not r.passed],
    }


async def run_all(cases=None) -> tuple[list[CaseResult], dict]:
    cases = cases if cases is not None else _load_dataset("benchmark_continuity").CASES
    results = [await run_case(c) for c in cases]
    return results, summarize(results)


def render_table(results: list[CaseResult], summary: dict) -> str:
    rows = ["| case | family | pass | missing | unlabelled | labelled | dup | request chars |",
            "|---|---|---|---|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r.name} | {r.family} | {'PASS' if r.passed else 'FAIL'} | "
                    f"{', '.join(r.missing) or '-'} | {', '.join(r.unlabelled) or '-'} | "
                    f"{', '.join(r.labelled) or '-'} | {r.duplicated_renders} | {r.request_chars} |")
    rows.append("")
    rows.append(f"survival_rate={summary['survival_rate']}  exclusion_rate={summary['exclusion_rate']}  "
                f"labelled_rate={summary['labelled_rate']}  duplicated_renders={summary['duplicated_renders']}  "
                f"cases={summary['cases']}")
    return "\n".join(rows)


def _git_sha() -> Optional[str]:
    try:
        import subprocess
        return subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def write_result(results: list[CaseResult], summary: dict, out_dir: Optional[Path] = None) -> Path:
    out_dir = out_dir or (_REPO_ROOT / "benchmark_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    sha = _git_sha() or "nosha"
    path = out_dir / f"continuity__{sha}__{ts}.json"
    payload = {
        "schema": 1, "kind": "context_delivery", "git_sha": sha, "run_ts": ts,
        "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME"),
        "summary": summary, "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    write = "--no-write" not in argv
    results, summary = asyncio.run(run_all())
    print(render_table(results, summary))
    if write:
        print(f"written: {write_result(results, summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
