"""The continuity set instrument (V3 Stage C) — runs on the dev box, no model.

These tests prove the INSTRUMENT: a real headless agent boots without
network, plants memories, asks a question, and the request it sends can be
scored. They do not assert that every case passes — survival and exclusion
rates are MEASUREMENTS (the Stage B gate compares before/after). Two cases
are asserted because they validate the instrument itself: the control fact
must survive, and the abstention marker must appear when nothing matches.
"""

from __future__ import annotations

import threading

import pytest

from blipshell.benchmark import continuity
from blipshell.benchmark.harness import _load_dataset

CASES = {c.name: c for c in _load_dataset("benchmark_continuity").CASES}


async def test_headless_agent_boots_without_network_or_background_tasks(tmp_path):
    agent, client = await continuity.bootstrap_headless_agent(tmp_path / "h.db")
    try:
        assert agent.sqlite is not None and agent.vectors is not None and agent.search is not None
        # background half never ran
        assert agent._memory_worker is not None and not agent._memory_worker.is_alive
        assert agent._health_check_task is None and agent._nightly_scheduler_task is None
        assert agent._embed_warmup_task is None
        assert all(ep.client is client for ep in agent.endpoint_manager._endpoints)
        # the fake embedder is wired into the REAL vector store
        v = agent.vectors._embed("hello")
        assert len(v) == agent.config.database.embedding_dimensions
        assert not any(t.name.startswith("MemoryWorker") for t in threading.enumerate())
    finally:
        await agent.sqlite.close()
        agent.vectors.close()


async def test_control_fact_survives_to_the_request():
    r = await continuity.run_case(CASES["control_short_fact"])
    assert r.passed, r
    assert r.request_chars > 0


async def test_abstention_marker_appears_when_nothing_matches():
    r = await continuity.run_case(CASES["abstention_marker_when_nothing_matches"])
    assert r.passed, r


async def test_chat_sends_exactly_one_request_per_turn(tmp_path):
    agent, client = await continuity.bootstrap_headless_agent(tmp_path / "one.db")
    try:
        await agent.start_session()
        client.sent.clear()
        reply = await agent.chat("hello there")
        assert reply == "ok"
        assert len(client.sent) == 1
        roles = [m["role"] for m in client.sent[0]]
        assert roles[0] == "system" and roles[-1] == "user"
    finally:
        await agent.sqlite.close()
        agent.vectors.close()


def test_score_request_labelling_rule():
    case = CASES["corrected_preference_current_question"]
    # unlabelled presence of the superseded fact = false recall
    bad = [{"role": "system", "content": "[12 days ago] I prefer tabs for indentation\n[2 days ago] I switched to spaces, four wide"}]
    r = continuity.score_request(case, bad)
    assert not r.passed and r.unlabelled == ["I prefer tabs"] and r.missing == []
    # the same text, labelled on its line, is acceptable evidence
    good = [{"role": "system", "content": "[12 days ago, superseded] I prefer tabs for indentation\n[2 days ago] I switched to spaces, four wide"}]
    r = continuity.score_request(case, good)
    assert r.passed and r.labelled == ["I prefer tabs"]


def test_history_question_requires_the_superseded_fact():
    case = CASES["corrected_preference_history_question"]
    only_new = [{"role": "system", "content": "[2 days ago] I switched to spaces, four wide"}]
    r = continuity.score_request(case, only_new)
    assert not r.passed and r.missing == ["I prefer tabs"]


async def test_run_all_reports_every_case_and_prints_the_table(capsys):
    results, summary = await continuity.run_all()
    assert len(results) == len(CASES)
    assert {r.name for r in results} == set(CASES)
    assert summary["cases"] == len(CASES)
    assert 0.0 <= summary["survival_rate"] <= 1.0
    assert 0.0 <= summary["exclusion_rate"] <= 1.0
    table = continuity.render_table(results, summary)
    print("\n" + table)  # the baseline readout, visible with -s
    assert "survival_rate=" in table


def test_dataset_is_well_formed():
    for c in CASES.values():
        assert c.family in ("survival", "false_recall"), c.name
        assert c.question and c.seeds, c.name
        assert c.must_appear or c.forbidden_unless_labelled or c.must_not_appear, c.name
        for s in c.seeds:
            assert s.role in ("user", "assistant") and s.kind in ("memory", "core", "lesson"), c.name
    names = [c.name for c in CASES.values()]
    assert len(names) == len(set(names))
