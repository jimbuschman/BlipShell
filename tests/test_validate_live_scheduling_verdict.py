"""The live scheduling validator's verdict: gate state at grant, not timestamps.

Real-corpus run 2026-09-16 (Ollama PC, data/live_scheduling_20260916_194004):
two BACKGROUND acquisitions were requested during a turn, parked 58.75 s and
46.86 s, and granted at 79.531 and 140.781 - the exact timestamps of the two
turn ends. The scheduler did its job. The validator called both failures,
because it judged "granted after the turn began" by `t_grant <= t_end`, and
`t_end` is stamped only after agent.chat() returns while the wrapper closes
the turn (waking the parked waiter) just before that return.

These replay those events through the corrected classification.
"""

from blipshell.llm.ollama_gate import BACKGROUND, INTERACTIVE
from scripts.validate_live_scheduling import classify_turn, verdict

TURN1 = (19.984, 79.531)
TURN2 = (79.531, 140.781)


def _ev(priority, t_req, t_grant, *, turn_at_grant, turn_at_request=True):
    return {"kind": "async", "priority": priority, "t_req": t_req, "t_grant": t_grant,
            "wait_ms": round((t_grant - t_req) * 1000, 1),
            "turn_at_request": turn_at_request, "turn_at_grant": turn_at_grant,
            "thread": "memory-worker" if priority == BACKGROUND else "MainThread"}


# The two background events exactly as recorded, with the gate state at grant
# the corrected instrument now records: the turn had already closed.
REAL_CORPUS_EVENTS = [
    _ev(INTERACTIVE, 20.5, 21.078, turn_at_grant=True),
    _ev(BACKGROUND, 20.781, 79.531, turn_at_grant=False),     # parked 58.75 s, woken at turn end
    _ev(INTERACTIVE, 79.6, 94.0, turn_at_grant=True),         # turn 2 waited behind that call
    _ev(BACKGROUND, 93.922, 140.781, turn_at_grant=False),    # parked 46.86 s, woken at turn end
]


def _report(turn_fields):
    return {
        "background_live_before_turns": True,
        "background_resumed_after_turns": True,
        "log_signals": {},
        "turns": [
            {**fields, "search_stats": {"chroma_hits": 59}, "contains_marker": True}
            for fields in turn_fields
        ],
    }


class TestGrantAtTurnEndIsNotAViolation:
    def test_real_corpus_events_pass(self):
        t1 = classify_turn(REAL_CORPUS_EVENTS, *TURN1)
        t2 = classify_turn(REAL_CORPUS_EVENTS, *TURN2)
        assert t1["background_grants_while_turn_open"] == 0
        assert t2["background_grants_while_turn_open"] == 0
        # ...and the parked-then-released calls are reported as such.
        assert t1["background_released_at_turn_end"] == 1
        assert t1["background_parked_ms"] == [58750.0]
        assert t2["background_released_at_turn_end"] == 1
        assert t2["background_parked_ms"] == [46859.0]
        # Turn 2's wait behind the call released at turn 1's end is visible.
        assert t2["background_grants_inflight_from_before_turn"] == 1
        assert verdict(_report([t1, t2])) == []

    def test_the_old_timestamp_rule_would_have_failed_these(self):
        """Documents the bug: by timestamps alone both grants sit inside
        their turn windows (t_grant == t_end)."""
        for (t_start, t_end), bg in ((TURN1, REAL_CORPUS_EVENTS[1]), (TURN2, REAL_CORPUS_EVENTS[3])):
            assert t_start <= bg["t_req"] and bg["t_grant"] <= t_end


class TestGrantWhileOpenIsAViolation:
    def test_background_granted_with_the_turn_open_is_a_miss(self):
        events = REAL_CORPUS_EVENTS + [
            _ev(BACKGROUND, 30.0, 45.0, turn_at_grant=True),  # got the gate mid-turn
        ]
        t1 = classify_turn(events, *TURN1)
        assert t1["background_grants_while_turn_open"] == 1
        misses = verdict(_report([t1, classify_turn(events, *TURN2)]))
        assert misses == ["turn 1: 1 background call(s) granted while the turn was open"]

    def test_verdict_still_names_the_other_contract_clauses(self):
        t1 = classify_turn(REAL_CORPUS_EVENTS, *TURN1)
        t2 = classify_turn(REAL_CORPUS_EVENTS, *TURN2)
        report = _report([t1, t2])
        report["turns"][1]["search_stats"] = {"chroma_hits": 0}
        report["turns"][1]["contains_marker"] = False
        report["background_resumed_after_turns"] = False
        report["log_signals"] = {"semantic_search_unavailable": 2}
        misses = verdict(report)
        assert misses == [
            "turn 2: no semantic hits (keyword fallback or empty vector search)",
            "turn 2: planted fact not recalled",
            "background work did not resume after the turns",
            "log: semantic_search_unavailable x2",
        ]
