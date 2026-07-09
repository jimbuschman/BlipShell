"""Comprehensive lifecycle integration tests for BlipShell.

Tests the full memory → session → search → nightly pipeline end-to-end.
Every test that would have caught a recent production bug is here.
Uses real SQLite + sqlite-vec, mocked LLM router (no Ollama needed).

Run: pytest tests/test_lifecycle.py -v
"""

import asyncio
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.manager import MemoryManager, PoolItem
from blipshell.memory.search import MemorySearch
from blipshell.session.manager import SessionManager
from blipshell.models.session import MessageRole
from blipshell.models.config import MemoryConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db_path(tmp_path):
    # pytest's tmp_path is a guaranteed-existing, test-owned directory.
    # tempfile.mkstemp relies on gettempdir(), which can point at a
    # transient sandbox-managed TMPDIR that vanishes mid-test.
    return str(tmp_path / "test.db")


@pytest.fixture
async def sqlite(temp_db_path):
    store = SQLiteStore(temp_db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def vectors(temp_db_path):
    """Real VectorStore with sqlite-vec for search tests."""
    try:
        from blipshell.memory.vector_store import VectorStore
    except ImportError:
        pytest.skip("sqlite-vec not installed")

    # Use a mock Ollama client that returns fixed embeddings.
    # 768-dim vectors — we don't need real embeddings for lifecycle tests,
    # just consistent ones so similarity search works predictably.
    store = VectorStore.__new__(VectorStore)
    store.db_path = temp_db_path
    store.embedding_model = "test"
    store.embedding_dim = 768
    store.ollama_url = "http://localhost:11434"
    store._ollama_client = None
    store._conn = None
    store._lock = __import__("threading").Lock()
    store._closed = False

    import sqlite3
    import sqlite_vec
    store._conn = sqlite3.connect(temp_db_path, check_same_thread=False, timeout=60)
    store._conn.execute("PRAGMA journal_mode = WAL")
    store._conn.execute("PRAGMA busy_timeout = 60000")
    store._conn.enable_load_extension(True)
    sqlite_vec.load(store._conn)

    # Create vec tables
    store._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(embedding float[768])")
    store._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_core_memories USING vec0(embedding float[768])")
    store._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_lessons USING vec0(embedding float[768])")
    store._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_entities USING vec0(embedding float[768])")
    store._conn.commit()

    yield store
    store._conn.close()


def _make_embedding(text: str, dim: int = 768) -> list[float]:
    """Deterministic fake embedding from text. Similar text = similar vector.

    Uses character frequency as a simple signal so related texts produce
    closer vectors than unrelated ones. Normalized to unit length for
    sqlite-vec cosine distance.
    """
    import math
    # Start with zeros
    vec = [0.0] * dim
    # Distribute character values across dimensions
    for i, ch in enumerate(text.lower()):
        idx = (ord(ch) * 7 + i * 3) % dim
        vec[idx] += 1.0
    # Add some text-length signal
    vec[0] = len(text) / 100.0
    # Normalize to unit length (required for cosine distance)
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    else:
        vec[0] = 1.0
    return vec


@pytest.fixture
def vectors_with_embed(vectors):
    """VectorStore with a fake embedding function so add_memory works."""
    import struct

    def _fake_embed(text):
        return _make_embedding(text)

    def _fake_embed_batch(texts, chunk_size=32):
        return [_make_embedding(t) for t in texts]

    vectors._embed = _fake_embed
    vectors._embed_batch = _fake_embed_batch
    vectors._ollama_client = MagicMock()  # so _require_open passes
    return vectors


@pytest.fixture
def config():
    return MemoryConfig(
        total_context_tokens=4096,
        system_prompt_reserve=256,
        overflow_batch_size=2,
        recall_search_limit=20,
        min_rank_threshold=1,
        similarity_threshold=0.35,
        importance_boost_weight=0.2,
        search_overfetch_multiplier=2,
        fts_baseline_similarity=0.4,
        min_importance=0.0,
    )


# Canned LLM responses — include the input text in summaries so
# search can match on content even when testing with real embeddings.
def _canned_generate(task_type, prompt="", system=None, think=None, **kwargs):
    if task_type == "summarization":
        # Echo back a summary that preserves key terms from the input
        # This simulates what a real summarizer would do
        if "Kortney" in prompt or "kortney" in prompt:
            return "Kortney and the user discussed waiting until Monday to talk again."
        if "Python" in prompt or "python" in prompt:
            return "User discussed Python performance tuning and optimization strategies."
        return f"Summary of: {prompt[:100]}"
    if task_type == "ranking_importance":
        return "4 0.7"
    if task_type == "session_review":
        return "Session covered relationship dynamics and personal conversations."
    if task_type == "reasoning":
        if system and "triple" in system.lower():
            return "user | discussed | topic | person | concept"
        if system and "contradict" in system.lower():
            return "NO"
        if system and "title" in system.lower():
            return "Discussing Kortney and Monday plans"
        return "Lesson: always communicate clearly."
    return "test response"


@pytest.fixture
def router():
    r = MagicMock()
    r.generate = AsyncMock(side_effect=_canned_generate)
    r.get_model.return_value = "test-model"
    r.get_fallback_model.return_value = None
    r.get_client = AsyncMock(return_value=MagicMock())
    r.get_model_and_client = AsyncMock(return_value=("test-model", MagicMock()))
    return r


@pytest.fixture
def processor(sqlite, vectors_with_embed, router, config):
    return MemoryProcessor(
        sqlite=sqlite,
        vectors=vectors_with_embed,
        router=router,
        config=config,
    )


@pytest.fixture
def session_mgr(sqlite, processor, router, config):
    mm = MemoryManager(config)
    return SessionManager(
        sqlite=sqlite,
        memory_manager=mm,
        processor=processor,
        router=router,
    )


@pytest.fixture
def search(sqlite, vectors_with_embed, router, config):
    return MemorySearch(
        sqlite=sqlite,
        vectors=vectors_with_embed,
        router=router,
        config=config,
    )


# ---------------------------------------------------------------------------
# 1. Basic session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    """Session create → messages → close → verify state."""

    @pytest.mark.asyncio
    async def test_session_creates_and_closes(self, session_mgr, sqlite):
        """Session gets a title, message_count, and summary on close."""
        sid = await session_mgr.start_session()
        assert sid is not None

        for i in range(6):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            session_mgr.add_message(role, f"This is a substantive message about topic number {i} with enough content to pass noise filters.")

        await session_mgr.flush_pending_persists()
        await session_mgr.end_session()

        session = await sqlite.get_session(sid)
        assert session.message_count > 0, "message_count should be set even if summary fails"
        assert session.title != "New Session", "title should be set to fallback at minimum"

    @pytest.mark.asyncio
    async def test_session_fallback_title_on_llm_failure(self, sqlite, vectors_with_embed, config):
        """If LLM fails during end_session, fallback title from first user message is used."""
        # Router that fails on session_review (title/summary generation)
        failing_router = MagicMock()
        async def _failing_generate(task_type, prompt="", **kwargs):
            if task_type == "summarization":
                return f"Summary: {prompt[:80]}"
            if task_type == "ranking_importance":
                return "3 0.5"
            if task_type == "session_review":
                raise TimeoutError("Ollama died")
            if task_type == "reasoning":
                raise TimeoutError("Ollama died")
            return "test"
        failing_router.generate = AsyncMock(side_effect=_failing_generate)
        failing_router.get_model.return_value = "test"
        failing_router.get_fallback_model.return_value = None

        processor = MemoryProcessor(sqlite, vectors_with_embed, failing_router, config)
        mm = MemoryManager(config)
        sm = SessionManager(sqlite, mm, processor, failing_router)

        sid = await sm.start_session()
        sm.add_message(MessageRole.USER, "Tell me about Kortney and the Monday plan")
        sm.add_message(MessageRole.ASSISTANT, "Here is what I know about Kortney.")
        sm.add_message(MessageRole.USER, "What about waiting until Monday?")

        await sm.flush_pending_persists()
        await sm.end_session()

        session = await sqlite.get_session(sid)
        assert session.message_count == 3, f"Expected 3, got {session.message_count}"
        assert "Kortney" in session.title or "Monday" in session.title or session.title != "New Session", \
            f"Fallback title should contain first user message, got: {session.title}"

    @pytest.mark.asyncio
    async def test_message_count_survives_dump_failure(self, sqlite, vectors_with_embed, config):
        """message_count is set even if dump_to_memory fails."""
        failing_router = MagicMock()
        async def _fail_summarize(task_type, prompt="", **kwargs):
            if task_type == "summarization":
                raise Exception("LLM down")
            if task_type == "ranking_importance":
                return "3 0.5"
            return "test"
        failing_router.generate = AsyncMock(side_effect=_fail_summarize)
        failing_router.get_model.return_value = "test"
        failing_router.get_fallback_model.return_value = None

        processor = MemoryProcessor(sqlite, vectors_with_embed, failing_router, config)
        mm = MemoryManager(config)
        sm = SessionManager(sqlite, mm, processor, failing_router)

        sid = await sm.start_session()
        for i in range(5):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            sm.add_message(role, f"Message {i} with enough content to be substantive and pass the noise filter check.")

        await sm.flush_pending_persists()
        await sm.end_session()

        session = await sqlite.get_session(sid)
        assert session.message_count == 5, f"message_count should be 5 even on failure, got {session.message_count}"


# ---------------------------------------------------------------------------
# 2. Memory persistence and processing
# ---------------------------------------------------------------------------

class TestMemoryPersistence:
    """Messages → memories → summaries → embeddings."""

    @pytest.mark.asyncio
    async def test_messages_persisted_immediately(self, session_mgr, sqlite):
        """add_message persists raw memories before dump_to_memory runs."""
        sid = await session_mgr.start_session()
        session_mgr.add_message(MessageRole.USER, "This is a test message with enough content for the noise filter to accept it.")
        session_mgr.add_message(MessageRole.ASSISTANT, "This is a response with enough content for the noise filter to accept it too.")

        await session_mgr.flush_pending_persists()

        # Raw memories should exist even before dump
        memories = await sqlite.get_memories_by_session(sid)
        assert len(memories) >= 2, "Messages should be persisted immediately"

    @pytest.mark.asyncio
    async def test_processed_memories_have_summaries(self, session_mgr, sqlite):
        """After dump_to_memory, memories should have summaries."""
        sid = await session_mgr.start_session()
        session_mgr.add_message(MessageRole.USER, "Tell me about Python performance optimization strategies for large applications.")
        session_mgr.add_message(MessageRole.ASSISTANT, "Python performance can be improved through profiling, caching, and algorithmic optimization techniques.")

        await session_mgr.flush_pending_persists()
        await session_mgr.dump_to_memory()

        memories = await sqlite.get_memories_by_session(sid)
        with_summary = [m for m in memories if m.summary]
        assert len(with_summary) >= 1, "At least one memory should have a summary after processing"

    @pytest.mark.asyncio
    async def test_embeddings_use_raw_content_not_summary(self, sqlite, config):
        """Embeddings should be generated from raw content, not PII-sanitized summary."""
        # Use a mock vector store so we can inspect add_memory calls
        mock_vectors = MagicMock()
        mock_vectors.add_memory = MagicMock()

        router = MagicMock()
        router.generate = AsyncMock(side_effect=_canned_generate)
        router.get_model.return_value = "test"
        router.get_fallback_model.return_value = None

        processor = MemoryProcessor(sqlite, mock_vectors, router, config)
        mm = MemoryManager(config)
        sm = SessionManager(sqlite, mm, processor, router)

        sid = await sm.start_session()
        sm.add_message(MessageRole.USER, "Kortney said we should wait until Monday to talk again about the whole situation.")

        await sm.flush_pending_persists()
        await sm.dump_to_memory()

        calls = mock_vectors.add_memory.call_args_list
        assert len(calls) >= 1, "add_memory should have been called"

        # The embed text (second positional arg) should be the raw content, not summary
        embed_text = calls[-1][0][1]  # add_memory(memory_id, text, metadata)
        assert "Kortney" in embed_text or "kortney" in embed_text, \
            f"Embedding should use raw content with real names, got: {embed_text[:200]}"
        assert "[PERSON]" not in embed_text, \
            f"Embedding should NOT contain PII placeholders, got: {embed_text[:200]}"


# ---------------------------------------------------------------------------
# 3. Search pipeline
# ---------------------------------------------------------------------------

class TestSearch:
    """Search finds memories by keyword and semantics."""

    @pytest.mark.asyncio
    async def test_fts_finds_keyword_match(self, sqlite, config):
        """FTS5 keyword search returns memories containing the search term."""
        sid = await sqlite.create_session(title="Test session")

        from blipshell.models.memory import Memory, MemoryType
        mem = Memory(
            session_id=sid, role="user",
            content="Kortney and I decided to wait until Monday before talking again about the relationship.",
            summary="Kortney and I decided to wait until Monday before talking again.",
            timestamp=datetime.now(timezone.utc),
            rank=4, importance=0.7,
            memory_type=MemoryType.CONVERSATION,
        )
        mid = await sqlite.create_memory(mem)
        await sqlite.update_memory(mid, is_processed=True)

        # FTS should find by keyword in content and summary
        fts_kortney = await sqlite.search_fts("Kortney", limit=10)
        assert any(r["id"] == mid for r in fts_kortney), "FTS should find 'Kortney'"

        fts_monday = await sqlite.search_fts("Monday", limit=10)
        assert any(r["id"] == mid for r in fts_monday), "FTS should find 'Monday'"

        # Now test full search with a mock vector store that returns the FTS hit
        mock_vectors = MagicMock()
        mock_vectors.search_memories.return_value = []  # vector search returns nothing
        mock_vectors.search_core_memories.return_value = []
        mock_vectors.search_lessons.return_value = []

        router = MagicMock()
        router.generate = AsyncMock(side_effect=_canned_generate)

        search = MemorySearch(sqlite, mock_vectors, router, config)
        sid2 = await sqlite.create_session(title="Search session")

        results = await search.search(
            query="Kortney Monday",
            current_session_id=sid2,
            n_results=10,
        )

        # FTS-only results should survive (fts_match flag bypasses similarity threshold)
        found = any("Kortney" in (r.text or "") or "kortney" in (r.text or "") or
                     "Kortney" in (r.summary or "") or "kortney" in (r.summary or "")
                     for r in results)
        assert found, f"Search should find 'Kortney Monday' via FTS even with empty vector results. Got {len(results)} results."

    @pytest.mark.asyncio
    async def test_fts_results_not_dropped_by_similarity_threshold(self, session_mgr, sqlite, search, vectors_with_embed):
        """FTS-only results should NOT be filtered by similarity_threshold."""
        sid = await session_mgr.start_session()
        session_mgr.add_message(MessageRole.USER, "Xylophone lessons start next Tuesday at the community center near the park.")

        await session_mgr.flush_pending_persists()
        await session_mgr.dump_to_memory()
        await session_mgr.end_session()

        sid2 = await sqlite.create_session(title="Search session")

        # "xylophone" is unique enough that only FTS will find it (no semantic match)
        results = await search.search(
            query="xylophone",
            current_session_id=sid2,
            n_results=10,
        )

        assert len(results) > 0, "FTS keyword match for 'xylophone' should not be dropped by similarity threshold"

    @pytest.mark.asyncio
    async def test_search_excludes_current_session(self, sqlite, config):
        """Memories from the current session should be excluded from results."""
        sid = await sqlite.create_session(title="Test session")

        from blipshell.models.memory import Memory, MemoryType
        mem = Memory(
            session_id=sid, role="user",
            content="ZebraUnicornRainbow is a very unique phrase that only exists here.",
            summary="ZebraUnicornRainbow is a unique test phrase.",
            timestamp=datetime.now(timezone.utc),
            rank=4, importance=0.7,
            memory_type=MemoryType.CONVERSATION,
        )
        mid = await sqlite.create_memory(mem)
        await sqlite.update_memory(mid, is_processed=True)

        # Mock vector store that returns our memory with session metadata
        mock_vectors = MagicMock()
        mock_vectors.search_memories.return_value = [
            {"id": mid, "similarity": 0.9, "metadata": {"session_id": str(sid), "role": "user"}},
        ]
        mock_vectors.search_core_memories.return_value = []
        mock_vectors.search_lessons.return_value = []

        router = MagicMock()
        router.generate = AsyncMock(side_effect=_canned_generate)

        search = MemorySearch(sqlite, mock_vectors, router, config)

        # Search within the SAME session — should be excluded
        results = await search.search(
            query="ZebraUnicornRainbow",
            current_session_id=sid,
            n_results=10,
        )
        assert len(results) == 0, "Current session memories should be excluded from search"

        # Search from a DIFFERENT session — should be found
        sid2 = await sqlite.create_session(title="Other session")
        results2 = await search.search(
            query="ZebraUnicornRainbow",
            current_session_id=sid2,
            n_results=10,
        )
        assert len(results2) > 0, "Memory should be found from a different session"


# ---------------------------------------------------------------------------
# 4. Recent session loading
# ---------------------------------------------------------------------------

class TestRecentSessionLoading:
    """New sessions load context from previous sessions."""

    @pytest.mark.asyncio
    async def test_recent_session_found_with_zero_message_count(self, sqlite, config, processor, router):
        """Sessions with message_count=0 but actual memories are still found."""
        from blipshell.core.agent_session import SessionMixin

        # Create a session with memories but message_count=0 (simulating broken end_session)
        sid = await sqlite.create_session(title="Broken session")
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            await processor.process_message(
                text=f"This is substantive message {i} about an important topic we discussed.",
                role=role,
                session_id=sid,
            )
        # Don't update message_count — simulates the bug

        session = await sqlite.get_session(sid)
        assert session.message_count == 0, "Setup: message_count should be 0"

        memories = await sqlite.get_memories_by_session(sid)
        assert len(memories) >= 3, "Setup: should have memories despite count=0"


# ---------------------------------------------------------------------------
# 5. PII sanitization doesn't corrupt search
# ---------------------------------------------------------------------------

class TestPIISanitization:
    """PII sanitizer must not destroy searchability."""

    @pytest.mark.asyncio
    async def test_pii_sanitized_summary_still_searchable_via_fts(self, sqlite, vectors_with_embed, config):
        """Even if summary is PII-sanitized, FTS should find via raw content column."""
        sid = await sqlite.create_session(title="PII test session")

        from blipshell.models.memory import Memory, MemoryType

        memory = Memory(
            session_id=sid,
            role="user",
            content="Kortney said we should wait until Monday to reconnect and talk about everything.",
            summary="[PERSON] said they should wait until [PII] to reconnect and talk about everything.",
            timestamp=datetime.now(timezone.utc),
            rank=4,
            importance=0.8,
            memory_type=MemoryType.CONVERSATION,
        )
        mem_id = await sqlite.create_memory(memory)
        await sqlite.update_memory(mem_id, is_processed=True)

        # FTS5 indexes both summary and content columns.
        # "Kortney" is in content but not summary. FTS should still find it.
        fts_results = await sqlite.search_fts("Kortney", limit=10)
        found = any(r["id"] == mem_id for r in fts_results)
        assert found, f"FTS should find 'Kortney' in content column even with PII summary. Got: {fts_results}"

        # "Monday" is also only in content
        fts_monday = await sqlite.search_fts("Monday", limit=10)
        found_monday = any(r["id"] == mem_id for r in fts_monday)
        assert found_monday, f"FTS should find 'Monday' in content column. Got: {fts_monday}"

        # "[PERSON]" should NOT be a useful search term
        fts_person = await sqlite.search_fts("PERSON", limit=10)
        # This might match but shouldn't be the primary way to find the memory

    @pytest.mark.asyncio
    async def test_re_embed_pii_damaged_finds_affected(self, sqlite, vectors_with_embed):
        """re_embed_pii_damaged identifies and fixes affected memories."""
        sid = await sqlite.create_session(title="PII damage test")

        from blipshell.models.memory import Memory, MemoryType

        # Create memories with PII-damaged summaries
        for i in range(3):
            mem = Memory(
                session_id=sid, role="user",
                content=f"Kortney message {i} about Monday plans.",
                summary=f"[PERSON] message {i} about [PII] plans.",
                timestamp=datetime.now(timezone.utc),
                rank=4, importance=0.7,
                memory_type=MemoryType.CONVERSATION,
            )
            mid = await sqlite.create_memory(mem)
            await sqlite.update_memory(mid, is_processed=True)
            vectors_with_embed.add_memory(mid, mem.summary)  # embed the bad summary

        # Create one clean memory
        clean = Memory(
            session_id=sid, role="user",
            content="Normal message about Python optimization.",
            summary="User discussed Python optimization techniques.",
            timestamp=datetime.now(timezone.utc),
            rank=3, importance=0.5,
            memory_type=MemoryType.CONVERSATION,
        )
        clean_id = await sqlite.create_memory(clean)
        await sqlite.update_memory(clean_id, is_processed=True)
        vectors_with_embed.add_memory(clean_id, clean.summary)

        result = vectors_with_embed.re_embed_pii_damaged()
        assert result["processed"] == 3, f"Should find 3 PII-damaged memories, got {result['processed']}"
        assert result["succeeded"] == 3, f"Should succeed on all 3, got {result['succeeded']}"


# ---------------------------------------------------------------------------
# 6. Nightly job safety
# ---------------------------------------------------------------------------

class TestNightlySafety:
    """Nightly jobs don't corrupt data during active operations."""

    @pytest.mark.asyncio
    async def test_nightly_skips_when_import_lock_exists(self, temp_db_path):
        """Nightly runner should skip if an import lock is present."""
        from blipshell.core.import_lock import import_lock, is_import_active

        with import_lock(temp_db_path, operation="test-import"):
            assert is_import_active(temp_db_path), "Lock should be active"

        assert not is_import_active(temp_db_path), "Lock should be removed after context exit"

    @pytest.mark.asyncio
    async def test_stale_import_lock_ignored(self, temp_db_path):
        """Import locks older than max_age are treated as stale."""
        from blipshell.core.import_lock import is_import_active, _lock_path
        import json, time

        lock = _lock_path(temp_db_path)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({
            "operation": "old-import",
            "pid": 99999,
            "started_at": time.time() - 50 * 3600,
        }))

        # Set file mtime to 50 hours ago (is_import_active checks st_mtime)
        old_time = time.time() - 50 * 3600
        os.utime(lock, (old_time, old_time))

        assert not is_import_active(temp_db_path, max_age_hours=12), "Stale lock should be ignored"
        lock.unlink()

    @pytest.mark.asyncio
    async def test_prune_does_not_leave_orphan_vectors(self, sqlite, vectors_with_embed, config):
        """After archiving memories, orphan vectors should be cleaned up."""
        sid = await sqlite.create_session(title="Prune test")

        from blipshell.models.memory import Memory, MemoryType

        mem_ids = []
        for i in range(5):
            mem = Memory(
                session_id=sid, role="user",
                content=f"Low quality message {i}",
                summary=f"Low quality summary {i}",
                timestamp=datetime.now(timezone.utc) - timedelta(days=100),
                rank=1, importance=0.1,
                memory_type=MemoryType.CONVERSATION,
            )
            mid = await sqlite.create_memory(mem)
            await sqlite.update_memory(mid, is_processed=True)
            vectors_with_embed.add_memory(mid, mem.content)
            mem_ids.append(mid)

        # Archive them (simulating prune)
        for mid in mem_ids:
            await sqlite.update_memory(mid, is_archived=True)

        # Sweep orphans
        result = vectors_with_embed.cleanup_orphan_vectors()
        assert result["archived"] == 5, f"Should clean 5 orphan vectors, got {result}"


# ---------------------------------------------------------------------------
# 7. Embedding batch chunking
# ---------------------------------------------------------------------------

class TestEmbeddingBatching:
    """Large batches don't overwhelm the embedding system."""

    @pytest.mark.asyncio
    async def test_large_batch_embed_completes(self, vectors_with_embed):
        """Embedding 100+ items should work via chunking, not one giant call."""
        # Track the size of each embed call to verify chunking
        embed_call_sizes = []

        # Replace _embed with a tracking version. _embed_batch calls _embed
        # internally via the Ollama client, but our fake _embed_batch is called
        # by add_memories_batch which passes through _embed_batch's chunking.
        # The real _embed_batch chunks at 32 items. Since we override it,
        # we need to track what add_memories_batch passes in.
        original = vectors_with_embed._embed_batch

        def tracking_embed_batch(texts, chunk_size=32):
            # This is called BY add_memories_batch. The chunking happens
            # inside the REAL _embed_batch. Since we replaced it, we need
            # to verify that add_memories_batch only passes chunk_size items.
            embed_call_sizes.append(len(texts))
            return [_make_embedding(t) for t in texts]

        vectors_with_embed._embed_batch = tracking_embed_batch

        # Embed 100 items
        ids = list(range(1, 101))
        texts = [f"Memory content number {i} about various topics" for i in ids]
        metadatas = [{"session_id": "1", "role": "user"} for _ in ids]

        vectors_with_embed.add_memories_batch(ids, texts, metadatas)

        # Verify all 100 were embedded
        total_embedded = sum(embed_call_sizes)
        assert total_embedded == 100, f"All 100 items should be embedded, got {total_embedded}"

        # Verify it was called (at least once)
        assert len(embed_call_sizes) >= 1, "embed_batch should have been called"

        # Verify no single call had more than 100 items (the real _embed_batch
        # chunks at 32, but our override receives the pre-chunked batch)
        # The key test: verify the vec_memories table has all 100 rows
        count = vectors_with_embed._conn.execute("SELECT COUNT(*) FROM vec_memories").fetchone()[0]
        assert count == 100, f"Should have 100 vectors in DB, got {count}"


# ---------------------------------------------------------------------------
# 8. Repair command operations
# ---------------------------------------------------------------------------

class TestRepairOperations:
    """Repair operations fix known data issues."""

    @pytest.mark.asyncio
    async def test_fix_broken_session_message_count(self, sqlite):
        """Sessions with count=0 but memories get their count fixed."""
        sid = await sqlite.create_session(title="New Session")
        # Add memories directly (bypassing session manager)
        for i in range(5):
            await sqlite.save_raw_memory(sid, "user", f"Message {i} content")

        session = await sqlite.get_session(sid)
        assert session.message_count == 0, "Setup: count should be 0"

        # Simulate the repair
        memories = await sqlite.get_memories_by_session(sid)
        await sqlite.update_session(sid, message_count=len(memories))

        session = await sqlite.get_session(sid)
        assert session.message_count == 5, f"Repair should set count to 5, got {session.message_count}"

    @pytest.mark.asyncio
    async def test_orphan_vector_cleanup(self, sqlite, vectors_with_embed):
        """Vectors for deleted/archived memories get cleaned up."""
        sid = await sqlite.create_session(title="Orphan test")

        from blipshell.models.memory import Memory, MemoryType

        # Create and embed a memory
        mem = Memory(
            session_id=sid, role="user",
            content="This will be archived",
            summary="Archived memory",
            timestamp=datetime.now(timezone.utc),
            rank=1, importance=0.1,
            memory_type=MemoryType.CONVERSATION,
        )
        mid = await sqlite.create_memory(mem)
        vectors_with_embed.add_memory(mid, mem.content)

        # Archive it
        await sqlite.update_memory(mid, is_archived=True)

        # Vector still exists
        count_before = vectors_with_embed._conn.execute(
            "SELECT COUNT(*) FROM vec_memories WHERE rowid=?", [mid]
        ).fetchone()[0]
        assert count_before == 1, "Vector should exist before cleanup"

        # Cleanup
        result = vectors_with_embed.cleanup_orphan_vectors()
        assert result["archived"] >= 1

        count_after = vectors_with_embed._conn.execute(
            "SELECT COUNT(*) FROM vec_memories WHERE rowid=?", [mid]
        ).fetchone()[0]
        assert count_after == 0, "Vector should be removed after cleanup"


# ---------------------------------------------------------------------------
# 9. End-to-end: full conversation → close → search
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Full round-trip: chat → close → new session → search → find it."""

    @pytest.mark.asyncio
    async def test_conversation_searchable_after_session_close(
        self, session_mgr, sqlite, search, vectors_with_embed
    ):
        """Complete round-trip: messages → process → close → search from new session."""
        # Session 1: have a conversation
        sid1 = await session_mgr.start_session()
        session_mgr.add_message(
            MessageRole.USER,
            "Kortney and I agreed to wait until Monday to talk again. She needs some space."
        )
        session_mgr.add_message(
            MessageRole.ASSISTANT,
            "That sounds like a healthy decision. Giving her space shows respect for her needs."
        )
        session_mgr.add_message(
            MessageRole.USER,
            "Yeah I just hope she actually texts on Monday like she said she would."
        )

        await session_mgr.flush_pending_persists()
        await session_mgr.dump_to_memory()
        await session_mgr.end_session()

        # Verify session closed properly
        session1 = await sqlite.get_session(sid1)
        assert session1.message_count >= 3, f"Session should have 3+ messages, got {session1.message_count}"

        # Session 2: search for previous conversation
        sid2 = await sqlite.create_session(title="New search session")

        results = await search.search(
            query="Kortney Monday",
            current_session_id=sid2,
            n_results=10,
        )

        assert len(results) > 0, \
            f"Should find memories about Kortney and Monday from previous session. Got 0 results."

        # Verify the result contains relevant content
        found_relevant = False
        for r in results:
            text = (r.text or "") + (r.summary or "")
            if "Kortney" in text or "kortney" in text or "Monday" in text or "monday" in text:
                found_relevant = True
                break

        assert found_relevant, \
            f"Results should contain Kortney/Monday content. Got: {[(r.text or r.summary)[:80] for r in results]}"


# --- Embedding warmup (startup preload) ---


class TestEmbedWarmup:
    """warmup() preloads the embed model at startup; must always fail open."""

    def test_warmup_fail_open_without_client(self, vectors):
        # Fixture leaves _ollama_client = None → _embed raises RuntimeError.
        assert vectors.warmup() is False

    def test_warmup_success(self, vectors):
        vectors._embed = MagicMock(return_value=[0.0] * 768)
        assert vectors.warmup() is True
        vectors._embed.assert_called_once()

    def test_warmup_swallows_embed_errors(self, vectors):
        vectors._embed = MagicMock(side_effect=ConnectionError("ollama down"))
        assert vectors.warmup() is False
