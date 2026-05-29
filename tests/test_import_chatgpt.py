"""Tests for ChatGPT import: JSONL/array parsing + dedup resume logic."""

import json
from unittest.mock import MagicMock

import pytest

from blipshell.import_chatgpt import parse_conversations
from blipshell.import_common import ParsedConversation, _resume_action


def _conv_record(conv_id: str, title: str, update_time: float, n_user_msgs: int = 1) -> dict:
    """Build a minimal ChatGPT-shaped conversation object."""
    mapping = {
        "root": {"id": "root", "message": None, "parent": None, "children": ["m1"]},
    }
    prev = "root"
    for i in range(n_user_msgs):
        node_id = f"m{i+1}"
        mapping[prev]["children"] = [node_id]
        mapping[node_id] = {
            "id": node_id,
            "parent": prev,
            "children": [],
            "message": {
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [f"message {i+1}"]},
                "create_time": 1775868000.0 + i,
            },
        }
        prev = node_id
    return {
        "title": title,
        "create_time": 1775868000.0,
        "update_time": update_time,
        "conversation_id": conv_id,
        "id": conv_id,
        "mapping": mapping,
        "current_node": prev,
    }


# -------- parsing --------


class TestParsing:
    def test_json_array(self, tmp_path):
        f = tmp_path / "conversations.json"
        f.write_text(json.dumps([
            _conv_record("c1", "First", 100.0),
            _conv_record("c2", "Second", 200.0),
        ]), encoding="utf-8")
        convs = parse_conversations(f)
        assert len(convs) == 2
        assert convs[0].title == "First"
        assert convs[0].external_id == "c1"
        assert convs[0].external_updated_at == 100.0

    def test_jsonl(self, tmp_path):
        f = tmp_path / "export.jsonl"
        lines = [
            json.dumps(_conv_record("c1", "First", 100.0)),
            json.dumps(_conv_record("c2", "Second", 200.0)),
        ]
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        convs = parse_conversations(f)
        assert len(convs) == 2
        assert {c.external_id for c in convs} == {"c1", "c2"}

    def test_jsonl_blank_lines_ignored(self, tmp_path):
        f = tmp_path / "export.jsonl"
        f.write_text(
            json.dumps(_conv_record("c1", "First", 100.0)) + "\n\n\n", encoding="utf-8"
        )
        convs = parse_conversations(f)
        assert len(convs) == 1

    def test_single_object(self, tmp_path):
        f = tmp_path / "one.json"
        f.write_text(json.dumps(_conv_record("c1", "Solo", 100.0)), encoding="utf-8")
        convs = parse_conversations(f)
        assert len(convs) == 1
        assert convs[0].external_id == "c1"

    def test_malformed_jsonl_line_raises(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text(
            json.dumps(_conv_record("c1", "First", 100.0)) + "\n{not json}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_conversations(f)

    def test_id_fallback_when_no_conversation_id(self, tmp_path):
        rec = _conv_record("c1", "First", 100.0)
        del rec["conversation_id"]  # only 'id' remains
        f = tmp_path / "x.jsonl"
        f.write_text(json.dumps(rec), encoding="utf-8")
        convs = parse_conversations(f)
        assert convs[0].external_id == "c1"


# -------- resume / dedup logic --------


@pytest.fixture
def mock_vectors():
    v = MagicMock()
    v.delete_memory = MagicMock()
    v.delete_lesson = MagicMock()
    return v


class TestResumeAction:
    async def test_new_conversation_imports(self, sqlite_store, mock_vectors):
        conv = ParsedConversation(title="Brand New", external_id="new1",
                                  external_updated_at=100.0)
        action = await _resume_action(sqlite_store, mock_vectors, conv)
        assert action == "import"

    async def test_unchanged_skips(self, sqlite_store, mock_vectors):
        sid = await sqlite_store.create_session(
            title="Existing", external_id="e1", external_updated_at=100.0,
        )
        await sqlite_store.update_session(sid, message_count=5)
        conv = ParsedConversation(title="Existing", external_id="e1",
                                  external_updated_at=100.0)
        action = await _resume_action(sqlite_store, mock_vectors, conv)
        assert action == "skip"

    async def test_grown_reimports(self, sqlite_store, mock_vectors):
        sid = await sqlite_store.create_session(
            title="Growing", external_id="g1", external_updated_at=100.0,
        )
        await sqlite_store.update_session(sid, message_count=5)
        # Export has a newer update_time → grown.
        conv = ParsedConversation(title="Growing", external_id="g1",
                                  external_updated_at=200.0)
        action = await _resume_action(sqlite_store, mock_vectors, conv)
        assert action == "reimport"
        # The old session should have been cleaned up.
        existing, count, _ = await sqlite_store.get_session_by_external_id("g1")
        assert existing is None

    async def test_incomplete_reimports(self, sqlite_store, mock_vectors):
        sid = await sqlite_store.create_session(
            title="Interrupted", external_id="i1", external_updated_at=100.0,
        )
        # message_count stays 0 → interrupted import.
        conv = ParsedConversation(title="Interrupted", external_id="i1",
                                  external_updated_at=100.0)
        action = await _resume_action(sqlite_store, mock_vectors, conv)
        assert action == "reimport"

    async def test_title_fallback_when_no_external_id(self, sqlite_store, mock_vectors):
        # Session created the old way (no external_id), conv has no external_id.
        sid = await sqlite_store.create_session(title="Legacy Chat")
        await sqlite_store.update_session(sid, message_count=3)
        conv = ParsedConversation(title="Legacy Chat", external_id=None)
        action = await _resume_action(sqlite_store, mock_vectors, conv)
        assert action == "skip"

    async def test_same_title_different_id_imports(self, sqlite_store, mock_vectors):
        # Two conversations share a generic title but differ by id —
        # the second must NOT be wrongly skipped.
        sid = await sqlite_store.create_session(
            title="New chat", external_id="first", external_updated_at=100.0,
        )
        await sqlite_store.update_session(sid, message_count=4)
        conv = ParsedConversation(title="New chat", external_id="second",
                                  external_updated_at=100.0)
        action = await _resume_action(sqlite_store, mock_vectors, conv)
        assert action == "import"
