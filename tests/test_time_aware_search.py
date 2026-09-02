"""Time-aware search (memory/timeparse.py + the search() partition).

The parser is deterministic on purpose (per-turn path, re-scorable forever),
so these tests pin exact ranges with a fixed `now` and explicit timezone.
The search-level tests drive the real MemorySearch with a mocked vector store
to prove the wiring: a query naming a time range ranks in-range memories
first, and everything degrades to the old ranking when the range is absent,
unrecognized, or empty.
"""

from datetime import datetime, timedelta, timezone

from blipshell.memory.timeparse import TimeRange, in_time_range, parse_time_range

# Fixed clock for every parser test: 2026-09-02 15:00 UTC, reasoning in UTC.
NOW = datetime(2026, 9, 2, 15, 0, tzinfo=timezone.utc)
UTC = timezone.utc


def parse(q):
    return parse_time_range(q, now=NOW, tz=UTC)


# --- day-level expressions ---------------------------------------------------

def test_yesterday():
    tr = parse("what did I say yesterday about the heron")
    assert tr.phrase == "yesterday"
    assert tr.start == datetime(2026, 9, 1, tzinfo=UTC)
    assert tr.end == datetime(2026, 9, 2, tzinfo=UTC)


def test_day_before_yesterday_not_swallowed_by_yesterday():
    tr = parse("the day before yesterday we discussed migrations")
    assert tr.phrase == "day before yesterday"
    assert tr.start == datetime(2026, 8, 31, tzinfo=UTC)
    assert tr.end == datetime(2026, 9, 1, tzinfo=UTC)


def test_today_variants():
    for phrase in ("today", "this morning", "this afternoon", "tonight",
                   "earlier today"):
        tr = parse(f"remind me what happened {phrase}")
        assert tr is not None, phrase
        assert tr.start == datetime(2026, 9, 2, tzinfo=UTC)
        assert tr.end == datetime(2026, 9, 3, tzinfo=UTC)


def test_last_night_spans_evening_to_morning():
    tr = parse("what was that idea from last night")
    assert tr.start == datetime(2026, 9, 1, 18, 0, tzinfo=UTC)
    assert tr.end == datetime(2026, 9, 2, 6, 0, tzinfo=UTC)


# --- relative windows --------------------------------------------------------

def test_in_the_last_n_days():
    tr = parse("what have we covered in the last 3 days")
    assert tr.start == NOW - timedelta(days=3)
    assert tr.end == NOW


def test_past_n_weeks_without_the():
    tr = parse("summarize the past 2 weeks")  # "the past 2 weeks" form
    assert tr.start == NOW - timedelta(weeks=2)
    assert tr.end == NOW


def test_n_days_ago_is_that_whole_day():
    tr = parse("what did I ask 3 days ago")
    assert tr.start == datetime(2026, 8, 30, tzinfo=UTC)
    assert tr.end == datetime(2026, 8, 31, tzinfo=UTC)


def test_hours_ago_is_a_window_not_an_instant():
    tr = parse("that thing from 2 hours ago")
    assert tr.start == NOW - timedelta(hours=3)
    assert tr.end == NOW - timedelta(hours=1)


def test_word_counts():
    tr = parse("what happened a couple days ago")
    assert tr.start == datetime(2026, 8, 31, tzinfo=UTC)


def test_recently_is_two_weeks():
    tr = parse("anything I mentioned recently?")
    assert tr.start == NOW - timedelta(days=14)
    assert tr.end == NOW


# --- calendar expressions ----------------------------------------------------

def test_last_month_is_the_previous_calendar_month():
    tr = parse("show me last month's discussions")
    assert tr.start == datetime(2026, 8, 1, tzinfo=UTC)
    assert tr.end == datetime(2026, 9, 1, tzinfo=UTC)


def test_this_and_last_week_are_monday_anchored():
    this_week = parse("what did we do this week")
    last_week = parse("what did we do last week")
    assert this_week.start.weekday() == 0
    assert last_week.start.weekday() == 0
    assert this_week.start - last_week.start == timedelta(days=7)
    assert last_week.end == this_week.start
    assert last_week.start <= NOW - timedelta(days=2)  # strictly before today


def test_named_month_resolves_to_most_recent_past_occurrence():
    tr = parse("what did I say in july")
    assert tr.start == datetime(2026, 7, 1, tzinfo=UTC)
    assert tr.end == datetime(2026, 8, 1, tzinfo=UTC)
    # A month later than now's month belongs to LAST year.
    tr = parse("back in november we planned this")
    assert tr.start == datetime(2025, 11, 1, tzinfo=UTC)


def test_named_month_with_year():
    tr = parse("in january 2025 I mentioned a book")
    assert tr.start == datetime(2025, 1, 1, tzinfo=UTC)
    assert tr.end == datetime(2025, 2, 1, tzinfo=UTC)


def test_bare_year():
    tr = parse("what were we doing back in 2024")
    assert tr.start == datetime(2024, 1, 1, tzinfo=UTC)
    assert tr.end == datetime(2025, 1, 1, tzinfo=UTC)


def test_weekday_is_most_recent_before_today():
    tr = parse("what did I tell you on monday")
    assert tr.start.weekday() == 0
    assert tr.end - tr.start == timedelta(days=1)
    assert NOW - timedelta(days=7) <= tr.start < datetime(2026, 9, 2, tzinfo=UTC)


# --- non-matches and range membership ---------------------------------------

def test_no_time_expression_returns_none():
    assert parse("tell me about the herons at the feeder") is None
    assert parse("what is my last name") is None  # "last" alone is not a range


def test_local_timezone_shifts_day_bounds():
    tz_plus10 = timezone(timedelta(hours=10))
    # 01:00 UTC = 11:00 local; "today" locally starts at 14:00 UTC yesterday.
    now = datetime(2026, 9, 2, 1, 0, tzinfo=UTC)
    tr = parse_time_range("what happened today", now=now, tz=tz_plus10)
    assert tr.start == datetime(2026, 9, 1, 14, 0, tzinfo=UTC)
    assert tr.end == datetime(2026, 9, 2, 14, 0, tzinfo=UTC)


def test_in_time_range_membership():
    tr = TimeRange(
        start=datetime(2026, 9, 1, tzinfo=UTC),
        end=datetime(2026, 9, 2, tzinfo=UTC),
        phrase="yesterday",
    )
    assert in_time_range(datetime(2026, 9, 1, 12, 0, tzinfo=UTC), tr)
    assert in_time_range(datetime(2026, 9, 1, 12, 0), tr)  # naive = UTC
    assert in_time_range(tr.start, tr)          # start inclusive
    assert not in_time_range(tr.end, tr)        # end exclusive
    assert not in_time_range(None, tr)


# --- search() wiring ---------------------------------------------------------

async def _two_memories(sqlite_store, mock_chroma):
    """One in-range memory (yesterday) and one strong out-of-range memory."""
    from blipshell.models.memory import Memory

    local = datetime.now(timezone.utc).astimezone()
    yesterday_noon = (local - timedelta(days=1)).replace(
        hour=12, minute=0, second=0, microsecond=0,
    ).astimezone(timezone.utc)

    a_id = await sqlite_store.create_memory(Memory(
        role="user", content="Saw the heron hunting by the river at lunch.",
        summary="Saw the heron hunting by the river.",
        timestamp=yesterday_noon, importance=0.8,
    ))
    b_id = await sqlite_store.create_memory(Memory(
        role="user", content="The heron nests near the old oak upstream.",
        summary="The heron nests near the old oak.",
        timestamp=datetime.now(timezone.utc) - timedelta(days=10),
        importance=0.8,
    ))
    # B is the semantically stronger candidate — without the time partition
    # it must win; with a "yesterday" query, A must.
    mock_chroma.search_memories.return_value = [
        {"id": b_id, "similarity": 0.9, "metadata": {}},
        {"id": a_id, "similarity": 0.6, "metadata": {}},
    ]
    return a_id, b_id


async def _search(sqlite_store, mock_chroma, mock_router, memory_config, query):
    from blipshell.memory.search import MemorySearch
    search = MemorySearch(sqlite_store, mock_chroma, mock_router,
                          config=memory_config)
    return search, await search.search(query)


async def test_time_query_ranks_in_range_first(
    sqlite_store, mock_chroma, mock_router, memory_config,
):
    a_id, b_id = await _two_memories(sqlite_store, mock_chroma)
    search, results = await _search(
        sqlite_store, mock_chroma, mock_router, memory_config,
        "what did I say about the heron yesterday",
    )
    assert [r.memory_id for r in results[:2]] == [a_id, b_id]
    stats = search.last_search_stats
    assert stats["time_range"]["phrase"] == "yesterday"
    assert stats["in_range_hits"] == 1


async def test_plain_query_keeps_score_order(
    sqlite_store, mock_chroma, mock_router, memory_config,
):
    a_id, b_id = await _two_memories(sqlite_store, mock_chroma)
    search, results = await _search(
        sqlite_store, mock_chroma, mock_router, memory_config,
        "tell me about the heron",
    )
    assert results[0].memory_id == b_id
    assert search.last_search_stats["time_range"] is None


async def test_empty_range_degrades_to_score_order(
    sqlite_store, mock_chroma, mock_router, memory_config,
):
    # A time phrase whose range contains neither memory: fail-soft, old order.
    a_id, b_id = await _two_memories(sqlite_store, mock_chroma)
    search, results = await _search(
        sqlite_store, mock_chroma, mock_router, memory_config,
        "what did I say about the heron in january 2020",
    )
    assert results[0].memory_id == b_id
    assert search.last_search_stats["in_range_hits"] == 0


async def test_switch_disables_parsing(
    sqlite_store, mock_chroma, mock_router, memory_config,
):
    memory_config.time_aware_search = False
    a_id, b_id = await _two_memories(sqlite_store, mock_chroma)
    search, results = await _search(
        sqlite_store, mock_chroma, mock_router, memory_config,
        "what did I say about the heron yesterday",
    )
    assert results[0].memory_id == b_id
    assert search.last_search_stats["time_range"] is None
