"""Tests for time awareness: the relative-time formatter, the get_current_time
tool, and the conversation-history stamping in _build_messages.

These are pure/deterministic — no LLM, runnable on the dev box. The behavioral
effect (does the model actually use the time signals) is validated separately
on the Ollama PC.
"""

from datetime import datetime, timedelta, timezone

from blipshell.core.agent_chat import format_relative_time
from blipshell.core.tools.time_tools import GetCurrentTimeTool


# ---------------------------------------------------------------------------
# format_relative_time
# ---------------------------------------------------------------------------

NOW = datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc)


def test_minutes_ago():
    assert format_relative_time(NOW - timedelta(minutes=20), NOW) == "[20m ago] "


def test_hours_ago():
    assert format_relative_time(NOW - timedelta(hours=3), NOW) == "[3h ago] "


def test_days_ago():
    assert format_relative_time(NOW - timedelta(days=2), NOW) == "[2d ago] "


def test_seven_plus_days_falls_back_to_absolute_date():
    ts = NOW - timedelta(days=10)
    assert format_relative_time(ts, NOW) == f"[{ts.strftime('%Y-%m-%d')}] "


def test_boundary_just_under_an_hour_is_minutes():
    assert format_relative_time(NOW - timedelta(minutes=59), NOW) == "[59m ago] "


def test_boundary_just_under_a_day_is_hours():
    assert format_relative_time(NOW - timedelta(hours=23), NOW) == "[23h ago] "


def test_falsy_timestamp_returns_empty():
    assert format_relative_time(None, NOW) == ""


def test_naive_timestamp_treated_as_utc():
    naive = (NOW - timedelta(hours=5)).replace(tzinfo=None)
    assert format_relative_time(naive, NOW) == "[5h ago] "


def test_label_has_trailing_space_for_prefixing():
    label = format_relative_time(NOW - timedelta(hours=1), NOW)
    assert label.endswith(" ")
    # Prefixing content should produce a clean "[1h ago] message" string.
    assert f"{label}hello" == "[1h ago] hello"


def test_now_defaults_to_current_time_when_omitted():
    # Just-created timestamp should read as minutes (0m), not error.
    label = format_relative_time(datetime.now(timezone.utc))
    assert label.endswith("ago] ")


# ---------------------------------------------------------------------------
# min_age_seconds suppression (kills "[0m ago]" noise on active-chat messages)
# ---------------------------------------------------------------------------

def test_min_age_suppresses_near_now_message():
    # A message 30s old must NOT be stamped when a 10-min floor is set.
    assert format_relative_time(NOW - timedelta(seconds=30), NOW, min_age_seconds=600) == ""


def test_min_age_suppresses_just_under_threshold():
    assert format_relative_time(NOW - timedelta(minutes=9), NOW, min_age_seconds=600) == ""


def test_min_age_allows_at_or_above_threshold():
    # 11 minutes old, 10-min floor — gap is real, so it stamps.
    assert format_relative_time(NOW - timedelta(minutes=11), NOW, min_age_seconds=600) == "[11m ago] "


def test_min_age_default_zero_stamps_everything():
    # Default behavior (memory rendering path) is unchanged: 0m still labeled.
    assert format_relative_time(NOW - timedelta(seconds=10), NOW) == "[0m ago] "


# ---------------------------------------------------------------------------
# GetCurrentTimeTool
# ---------------------------------------------------------------------------

def test_tool_definition():
    tool = GetCurrentTimeTool()
    defn = tool.definition()
    assert defn.name == "get_current_time"
    assert defn.parameters == []


def test_tool_is_read_only():
    # Must be safe in plan mode.
    assert GetCurrentTimeTool().read_only is True


async def test_tool_execute_returns_local_and_utc():
    result = await GetCurrentTimeTool().execute()
    assert "Local:" in result
    assert "UTC:" in result
    assert "ISO:" in result
