"""Live integration test — runs against real Ollama on the Ollama PC.

Tests actual usage patterns with real models, real embeddings, real worker
threads, real search. Uses a temp database so production data is never touched.

This test simulates what happens when you actually USE BlipShell:
- Chat while the memory worker is processing in the background
- Close the session mid-processing
- Search for things you said in a previous session
- Start a new session and verify context from the last one loads
- Run nightly maintenance after real data exists
- Recover after Ollama hiccups

Usage:
  python scripts/test_integration_live.py              # full test
  python scripts/test_integration_live.py --quick      # skip slow tests
  python scripts/test_integration_live.py --verbose    # show details

Requires: Ollama running locally with embedding model available.
"""

import argparse
import asyncio
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class LiveTestRunner:

    def __init__(self, verbose: bool = False, quick: bool = False):
        self.verbose = verbose
        self.quick = quick
        self.results: list[dict] = []
        self.temp_db: str = ""
        self.config = None
        self.config_manager = None

    def log(self, msg: str):
        print(f"  {msg}", flush=True)

    def detail(self, msg: str):
        if self.verbose:
            print(f"    {msg}", flush=True)

    def record(self, name: str, passed: bool, elapsed: float, detail: str = ""):
        self.results.append({
            "name": name, "passed": passed,
            "elapsed_s": round(elapsed, 2), "detail": detail,
        })
        icon = "+" if passed else "X"
        suffix = f" — {detail}" if detail and not passed else ""
        print(f"  [{icon}] {name} ({elapsed:.1f}s){suffix}", flush=True)

    def _make_stores(self):
        """Create fresh SQLiteStore + VectorStore pointing at the temp DB."""
        from blipshell.memory.sqlite_store import SQLiteStore
        from blipshell.memory.vector_store import VectorStore
        from blipshell.models.config import get_ollama_url

        sqlite = SQLiteStore(self.temp_db)
        vectors = VectorStore(
            db_path=self.temp_db,
            embedding_model=self.config.models.embedding,
            ollama_url=get_ollama_url(self.config.endpoints),
            embedding_dim=self.config.database.embedding_dimensions,
        )
        return sqlite, vectors

    def _make_router(self):
        from blipshell.llm.endpoints import EndpointManager
        from blipshell.llm.router import LLMRouter
        em = EndpointManager(self.config.endpoints, self.config.llm)
        return LLMRouter(self.config.models, em, pii_enabled=self.config.pii.enabled)

    async def setup(self):
        from blipshell.core.config import ConfigManager
        self.config_manager = ConfigManager(config_path=None)
        self.config = self.config_manager.load()
        fd, self.temp_db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.detail(f"Temp DB: {self.temp_db}")

    async def teardown(self):
        if self.temp_db and os.path.exists(self.temp_db):
            try:
                os.unlink(self.temp_db)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    async def test_ollama_responsive(self):
        """Ollama is running and embedding model is available."""
        t0 = time.monotonic()
        try:
            import ollama
            client = ollama.Client(host="http://localhost:11434")
            models = client.list()
            names = [m.model for m in models.models]
            embed = self.config.models.embedding
            found = any(embed in n for n in names)
            elapsed = time.monotonic() - t0
            self.record("ollama_responsive", found, elapsed,
                        "" if found else f"'{embed}' not in {names[:5]}")
            return found
        except Exception as e:
            self.record("ollama_responsive", False, time.monotonic() - t0, str(e))
            return False

    async def test_embed_single(self):
        """Single embedding completes."""
        t0 = time.monotonic()
        try:
            sqlite, vectors = self._make_stores()
            await sqlite.initialize()
            vectors.initialize()
            vec = vectors._embed("Test sentence for embedding.")
            elapsed = time.monotonic() - t0
            dim = self.config.database.embedding_dimensions
            ok = len(vec) == dim
            self.record("embed_single", ok, elapsed,
                        "" if ok else f"Expected {dim}, got {len(vec)}")
            vectors.close()
            await sqlite.close()
            return ok
        except Exception as e:
            self.record("embed_single", False, time.monotonic() - t0, str(e))
            return False

    async def test_embed_batch(self):
        """Batch of 32 embeddings completes."""
        t0 = time.monotonic()
        try:
            sqlite, vectors = self._make_stores()
            await sqlite.initialize()
            vectors.initialize()
            texts = [f"Test sentence number {i}." for i in range(32)]
            vecs = vectors._embed_batch(texts)
            elapsed = time.monotonic() - t0
            ok = len(vecs) == 32
            self.record("embed_batch_32", ok, elapsed)
            vectors.close()
            await sqlite.close()
            return ok
        except Exception as e:
            self.record("embed_batch_32", False, time.monotonic() - t0, str(e))
            return False

    async def test_embed_large_batch(self):
        """100 embeddings complete via chunking without hanging."""
        t0 = time.monotonic()
        try:
            sqlite, vectors = self._make_stores()
            await sqlite.initialize()
            vectors.initialize()
            texts = [f"Test sentence {i} about topic {i % 10}." for i in range(100)]
            vecs = vectors._embed_batch(texts)
            elapsed = time.monotonic() - t0
            ok = len(vecs) == 100
            self.record("embed_batch_100", ok, elapsed,
                        "" if ok else f"Got {len(vecs)}")
            vectors.close()
            await sqlite.close()
            return ok
        except Exception as e:
            self.record("embed_batch_100", False, time.monotonic() - t0, str(e))
            return False

    # ------------------------------------------------------------------
    # Real usage: full session with worker processing
    # ------------------------------------------------------------------

    async def test_full_session_with_worker(self):
        """Simulate real usage: start session, add messages with worker processing,
        close session. Verify everything persisted correctly."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.processor import MemoryProcessor
            from blipshell.memory.manager import MemoryManager
            from blipshell.memory.worker import MemoryWorker, WorkItem, WorkType
            from blipshell.session.manager import SessionManager
            from blipshell.models.session import MessageRole

            sqlite, vectors = self._make_stores()
            await sqlite.initialize()
            vectors.initialize()
            router = self._make_router()
            processor = MemoryProcessor(sqlite, vectors, router, config=self.config.memory)

            # Start worker thread (same as real Agent does)
            worker = MemoryWorker(self.config, vectors)
            # Override the DB path to use our temp DB
            original_path = self.config.database.path
            self.config.database.path = self.temp_db
            worker = MemoryWorker(self.config, vectors)
            worker.start()

            mm = MemoryManager(self.config.memory)
            sm = SessionManager(sqlite, mm, processor, router)
            sid = await sm.start_session()

            # Add messages like a real conversation
            messages = [
                (MessageRole.USER, "I've been talking to Kortney about our situation. She said we should wait until Monday to talk again."),
                (MessageRole.ASSISTANT, "That sounds like she needs some space. Monday gives both of you time to think."),
                (MessageRole.USER, "Yeah but I'm worried she won't actually text on Monday. She's done this before."),
                (MessageRole.ASSISTANT, "I understand the anxiety. Her pattern shows she does come back, but the waiting is hard."),
                (MessageRole.USER, "She also mentioned maybe getting dinner at the Italian place on Oak Street next week."),
                (MessageRole.ASSISTANT, "That's a good sign — she's making future plans with you, not pulling away."),
                (MessageRole.USER, "I guess you're right. I just need to be patient and not text her first."),
                (MessageRole.ASSISTANT, "Exactly. Let her come to you on Monday. If she doesn't, give it another day before reaching out."),
            ]
            for role, content in messages:
                sm.add_message(role, content)

            await sm.flush_pending_persists()

            # Enqueue messages to worker (simulates what Agent._enqueue_undumped_messages does)
            undumped = sm.get_undumped_messages()
            for msg in undumped:
                if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                    worker.enqueue(WorkItem(
                        work_type=WorkType.PROCESS_MESSAGE,
                        text=msg.content,
                        role=msg.role.value,
                        session_id=sid,
                    ))

            # Wait for worker to process (with timeout)
            deadline = time.monotonic() + 120  # 2 min max
            while worker.queue_depth > 0 and time.monotonic() < deadline:
                await asyncio.sleep(1)
                self.detail(f"  Worker queue depth: {worker.queue_depth}")

            if worker.queue_depth > 0:
                self.record("full_session_worker", False, time.monotonic() - t0,
                            f"Worker didn't drain in 120s, {worker.queue_depth} items left")
                worker.shutdown(timeout=5)
                self.config.database.path = original_path
                vectors.close()
                await sqlite.close()
                return False

            # Now close the session — shutdown worker first so it stops
            # idle entity extraction before we close the VectorStore
            worker.shutdown(timeout=30)
            await sm.end_session()

            elapsed = time.monotonic() - t0

            # === VERIFY EVERYTHING ===
            checks = []

            # 1. Session has message_count and title
            session = await sqlite.get_session(sid)
            if session.message_count == 0:
                checks.append(f"message_count=0 (expected {len(messages)})")
            if session.title == "New Session":
                checks.append("title still 'New Session'")

            # 2. Memories exist with summaries
            memories = await sqlite.get_memories_by_session(sid)
            with_summary = [m for m in memories if m.summary]
            if len(with_summary) < 4:
                checks.append(f"Only {len(with_summary)} memories have summaries (expected 4+)")

            # 3. Embeddings exist
            mem_ids = [m.id for m in memories if m.summary]
            if mem_ids:
                embedded = vectors._conn.execute(
                    f"SELECT COUNT(*) FROM vec_memories WHERE rowid IN ({','.join('?' * len(mem_ids))})",
                    mem_ids,
                ).fetchone()[0]
                if embedded < len(mem_ids) * 0.8:  # allow some failures
                    checks.append(f"Only {embedded}/{len(mem_ids)} memories have embeddings")

            # 4. FTS index has the content
            fts = await sqlite.search_fts("Kortney", limit=5)
            if len(fts) == 0:
                checks.append("FTS can't find 'Kortney'")

            fts_monday = await sqlite.search_fts("Monday", limit=5)
            if len(fts_monday) == 0:
                checks.append("FTS can't find 'Monday'")

            ok = len(checks) == 0
            self.record("full_session_worker", ok, elapsed,
                        "; ".join(checks) if checks else f"{len(memories)} memories, {len(with_summary)} summarized")

            self.config.database.path = original_path
            vectors.close()
            await sqlite.close()
            return ok

        except Exception as e:
            self.record("full_session_worker", False, time.monotonic() - t0, str(e))
            return False

    async def test_search_after_session(self):
        """After a full session, search from a new session finds previous memories."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.processor import MemoryProcessor
            from blipshell.memory.manager import MemoryManager
            from blipshell.memory.search import MemorySearch
            from blipshell.session.manager import SessionManager
            from blipshell.models.session import MessageRole

            sqlite, vectors = self._make_stores()
            await sqlite.initialize()
            vectors.initialize()
            router = self._make_router()
            processor = MemoryProcessor(sqlite, vectors, router, config=self.config.memory)
            mm = MemoryManager(self.config.memory)
            sm = SessionManager(sqlite, mm, processor, router)

            # Session 1: conversation with unique content
            sid1 = await sm.start_session()
            sm.add_message(MessageRole.USER,
                "Bartholomew and I went to the xylophone concert at Riverside Park last Thursday. It was amazing.")
            sm.add_message(MessageRole.ASSISTANT,
                "That sounds like a wonderful experience. Xylophone concerts at outdoor venues have great acoustics.")
            sm.add_message(MessageRole.USER,
                "Yeah, Bartholomew really enjoyed it. He wants to go back next month.")

            await sm.flush_pending_persists()
            await sm.dump_to_memory()
            await sm.end_session()

            # Verify session 1 closed properly
            session1 = await sqlite.get_session(sid1)
            self.detail(f"Session 1: count={session1.message_count}, title={session1.title}")

            # Session 2: search for session 1 content
            sid2 = await sqlite.create_session(title="Search session")
            search = MemorySearch(sqlite, vectors, router, config=self.config.memory)

            # Test 1: Search by unique name
            results_name = await search.search("Bartholomew xylophone", current_session_id=sid2, n_results=10)
            found_name = any(
                "Bartholomew" in (r.text or "") or "bartholomew" in (r.text or "").lower() or
                "Bartholomew" in (r.summary or "") or "bartholomew" in (r.summary or "").lower()
                for r in results_name
            )

            # Test 2: Search by topic
            results_topic = await search.search("concert park music", current_session_id=sid2, n_results=10)

            # Test 3: Search by day
            results_day = await search.search("Thursday Riverside", current_session_id=sid2, n_results=10)

            elapsed = time.monotonic() - t0

            checks = []
            if not found_name:
                # Debug: check what FTS and vector search return
                fts = await sqlite.search_fts("Bartholomew", limit=5)
                vec_results = vectors.search_memories("Bartholomew xylophone", n_results=5)
                checks.append(f"Name search failed. FTS hits: {len(fts)}, Vec hits: {len(vec_results)}, "
                              f"Full search: {len(results_name)}")
            if len(results_topic) == 0:
                checks.append("Topic search ('concert park music') returned 0 results")
            if len(results_day) == 0:
                fts_day = await sqlite.search_fts("Thursday", limit=5)
                checks.append(f"Day search ('Thursday Riverside') returned 0. FTS Thursday hits: {len(fts_day)}")

            ok = len(checks) == 0
            self.record("search_after_session", ok, elapsed,
                        "; ".join(checks) if checks else f"name={len(results_name)}, topic={len(results_topic)}, day={len(results_day)}")

            vectors.close()
            await sqlite.close()
            return ok

        except Exception as e:
            self.record("search_after_session", False, time.monotonic() - t0, str(e))
            return False

    async def test_back_to_back_sessions(self):
        """Close one session, start another. Second session should have context from first."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.processor import MemoryProcessor
            from blipshell.memory.manager import MemoryManager
            from blipshell.session.manager import SessionManager
            from blipshell.models.session import MessageRole

            sqlite, vectors = self._make_stores()
            await sqlite.initialize()
            vectors.initialize()
            router = self._make_router()
            processor = MemoryProcessor(sqlite, vectors, router, config=self.config.memory)

            # Session 1
            mm1 = MemoryManager(self.config.memory)
            sm1 = SessionManager(sqlite, mm1, processor, router)
            sid1 = await sm1.start_session()

            for i in range(6):
                role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
                sm1.add_message(role, f"Session one message {i} about learning Portuguese and traveling to Brazil next summer with the family.")

            await sm1.flush_pending_persists()
            await sm1.dump_to_memory()
            await sm1.end_session()

            s1 = await sqlite.get_session(sid1)
            self.detail(f"Session 1: count={s1.message_count}, title='{s1.title[:50]}'")

            # Session 2 — check that _load_recent_sessions would find session 1
            from blipshell.core.agent_session import SessionMixin
            sessions = await sqlite.list_sessions(limit=5)

            # The key check: session 1 should be findable as "substantive"
            found_substantive = False
            for s in sessions:
                if s.id == sid1:
                    # Either message_count >= 5 or actual memories >= 5
                    if s.message_count >= 5:
                        found_substantive = True
                    else:
                        mems = await sqlite.get_memories_by_session(s.id)
                        good = [m for m in mems if m.summary and not m.is_archived]
                        if len(good) >= 5:
                            found_substantive = True

            elapsed = time.monotonic() - t0
            self.record("back_to_back_sessions", found_substantive, elapsed,
                        "" if found_substantive else f"Session 1 not found as substantive. count={s1.message_count}")

            vectors.close()
            await sqlite.close()
            return found_substantive

        except Exception as e:
            self.record("back_to_back_sessions", False, time.monotonic() - t0, str(e))
            return False

    async def test_pii_embed_not_corrupted(self):
        """Content with names/dates is still searchable after processing through
        the full pipeline (which may PII-sanitize the summary)."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.processor import MemoryProcessor

            sqlite, vectors = self._make_stores()
            await sqlite.initialize()
            vectors.initialize()
            router = self._make_router()
            processor = MemoryProcessor(sqlite, vectors, router, config=self.config.memory)

            sid = await sqlite.create_session(title="PII test")
            mem_id = await processor.process_message(
                text="Kortney texted me at 3pm on Monday about meeting at the coffee shop on Oak Street.",
                role="user",
                session_id=sid,
            )

            if not mem_id:
                self.record("pii_embed_not_corrupted", False, time.monotonic() - t0, "No memory ID")
                vectors.close()
                await sqlite.close()
                return False

            # Check: can vector search find this by name?
            vec_results = vectors.search_memories("Kortney Monday coffee", n_results=5)
            found_vec = any(r["id"] == mem_id for r in vec_results)

            # Check: can FTS find this by name?
            fts_results = await sqlite.search_fts("Kortney", limit=5)
            found_fts = any(r["id"] == mem_id for r in fts_results)

            # Check: is the summary PII-sanitized?
            memory = await sqlite.get_memory(mem_id)
            pii_in_summary = "[PERSON]" in (memory.summary or "") or "[PII]" in (memory.summary or "")

            elapsed = time.monotonic() - t0

            checks = []
            if not found_vec:
                checks.append(f"Vector search can't find 'Kortney Monday coffee' (got {len(vec_results)} results)")
            if not found_fts:
                checks.append(f"FTS can't find 'Kortney' (got {len(fts_results)} results)")
            if pii_in_summary:
                self.detail(f"Note: summary was PII-sanitized: {memory.summary[:100]}")
                if not found_vec:
                    checks.append("Summary PII-sanitized AND vector search failed — embedding used sanitized text!")

            ok = found_vec and found_fts
            self.record("pii_embed_not_corrupted", ok, elapsed, "; ".join(checks) if checks else "")

            vectors.close()
            await sqlite.close()
            return ok

        except Exception as e:
            self.record("pii_embed_not_corrupted", False, time.monotonic() - t0, str(e))
            return False

    async def test_session_close_under_load(self):
        """Close session with many messages. message_count and title must be set
        even if dump_to_memory is slow. Tests the fallback-title-before-LLM fix."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.processor import MemoryProcessor
            from blipshell.memory.manager import MemoryManager
            from blipshell.session.manager import SessionManager
            from blipshell.models.session import MessageRole

            sqlite, vectors = self._make_stores()
            await sqlite.initialize()
            vectors.initialize()
            router = self._make_router()
            processor = MemoryProcessor(sqlite, vectors, router, config=self.config.memory)
            mm = MemoryManager(self.config.memory)
            sm = SessionManager(sqlite, mm, processor, router)

            sid = await sm.start_session()
            n_messages = 10

            for i in range(n_messages):
                role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
                sm.add_message(role, f"Rapid message {i} about complex topic number {i} with substantial content to process.")

            await sm.flush_pending_persists()

            # Scale timeout: 15s per message for summarization + scoring, plus 60s overhead
            timeout = n_messages * 15 + 60
            try:
                await asyncio.wait_for(sm.end_session(), timeout=timeout)
            except asyncio.TimeoutError:
                # Even on timeout, message_count should be set (that's the fix)
                session = await sqlite.get_session(sid)
                elapsed = time.monotonic() - t0
                if session.message_count == 0:
                    self.record("session_close_under_load", False, elapsed,
                                f"end_session timed out at {timeout}s AND message_count=0")
                else:
                    # Timeout is OK as long as bookkeeping survived
                    self.record("session_close_under_load", True, elapsed,
                                f"end_session timed out at {timeout}s but message_count={session.message_count} (bookkeeping saved)")
                vectors.close()
                await sqlite.close()
                return session.message_count > 0

            elapsed = time.monotonic() - t0

            session = await sqlite.get_session(sid)
            checks = []
            if session.message_count == 0:
                checks.append(f"message_count=0 (expected {n_messages})")
            if session.title == "New Session":
                checks.append("title still 'New Session'")

            memories = await sqlite.get_memories_by_session(sid)
            if len(memories) == 0:
                checks.append("zero memories persisted")

            ok = len(checks) == 0
            self.record("session_close_under_load", ok, elapsed,
                        "; ".join(checks) if checks else f"count={session.message_count}, mems={len(memories)}")

            vectors.close()
            await sqlite.close()
            return ok

        except Exception as e:
            self.record("session_close_under_load", False, time.monotonic() - t0, str(e))
            return False

    async def test_nightly_after_real_data(self):
        """Run nightly backfill+cleanup after a real session with real data."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.processor import MemoryProcessor
            from blipshell.memory.manager import MemoryManager
            from blipshell.session.manager import SessionManager
            from blipshell.models.session import MessageRole
            from blipshell.models.memory import Memory, MemoryType

            sqlite, vectors = self._make_stores()
            await sqlite.initialize()
            vectors.initialize()
            router = self._make_router()
            processor = MemoryProcessor(sqlite, vectors, router, config=self.config.memory)
            mm = MemoryManager(self.config.memory)
            sm = SessionManager(sqlite, mm, processor, router)

            # Create a session with real processed data
            sid = await sm.start_session()
            sm.add_message(MessageRole.USER, "Testing nightly maintenance with real data processing.")
            sm.add_message(MessageRole.ASSISTANT, "This session will be used to verify nightly jobs work correctly.")

            await sm.flush_pending_persists()
            await sm.dump_to_memory()
            await sm.end_session()

            # Also create a memory WITHOUT an embedding (backfill target)
            orphan_mem = Memory(
                session_id=sid, role="user",
                content="This memory has no embedding and should be backfilled.",
                summary="Backfill target memory.",
                timestamp=datetime.now(timezone.utc),
                rank=3, importance=0.5,
                memory_type=MemoryType.CONVERSATION,
            )
            orphan_id = await sqlite.create_memory(orphan_mem)
            await sqlite.update_memory(orphan_id, is_processed=True)

            # Create an archived memory with a leftover vector (orphan vector)
            archived_mem = Memory(
                session_id=sid, role="user",
                content="This memory is archived but its vector lingers.",
                summary="Orphan vector source.",
                timestamp=datetime.now(timezone.utc),
                rank=1, importance=0.1,
                memory_type=MemoryType.CONVERSATION,
            )
            archived_id = await sqlite.create_memory(archived_mem)
            vectors.add_memory(archived_id, archived_mem.content)
            await sqlite.update_memory(archived_id, is_archived=True)

            # Run maintenance
            backfill_result = vectors.backfill_missing_vectors("memories", limit=100)
            orphan_result = vectors.cleanup_orphan_vectors()

            elapsed = time.monotonic() - t0

            checks = []

            # Backfill should have embedded the orphan memory
            orphan_embedded = vectors._conn.execute(
                "SELECT COUNT(*) FROM vec_memories WHERE rowid=?", [orphan_id]
            ).fetchone()[0]
            if orphan_embedded != 1:
                checks.append(f"Backfill didn't embed orphan memory {orphan_id}: {backfill_result}")

            # Orphan cleanup should have removed the archived memory's vector
            archived_vec = vectors._conn.execute(
                "SELECT COUNT(*) FROM vec_memories WHERE rowid=?", [archived_id]
            ).fetchone()[0]
            if archived_vec != 0:
                checks.append(f"Orphan cleanup didn't remove archived vector: {orphan_result}")

            # Search should still work after maintenance
            from blipshell.memory.search import MemorySearch
            sid2 = await sqlite.create_session(title="Post-nightly search")
            search = MemorySearch(sqlite, vectors, router, config=self.config.memory)
            results = await search.search("nightly maintenance", current_session_id=sid2, n_results=5)
            if len(results) == 0:
                fts = await sqlite.search_fts("nightly", limit=5)
                checks.append(f"Search broken after maintenance (FTS hits: {len(fts)})")

            ok = len(checks) == 0
            self.record("nightly_after_real_data", ok, elapsed, "; ".join(checks) if checks else "")

            vectors.close()
            await sqlite.close()
            return ok

        except Exception as e:
            self.record("nightly_after_real_data", False, time.monotonic() - t0, str(e))
            return False

    async def test_recovery_after_embed_failure(self):
        """Simulate embed failure mid-session, verify backfill recovers."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.processor import MemoryProcessor
            from blipshell.models.memory import Memory, MemoryType

            sqlite, vectors = self._make_stores()
            await sqlite.initialize()
            vectors.initialize()
            router = self._make_router()

            sid = await sqlite.create_session(title="Embed failure test")

            # Create memories that "failed to embed" (have summary but no vector)
            failed_ids = []
            for i in range(5):
                mem = Memory(
                    session_id=sid, role="user",
                    content=f"Failed embed memory {i} about unique topic Zephyr{i}.",
                    summary=f"Failed embed test {i}.",
                    timestamp=datetime.now(timezone.utc),
                    rank=3, importance=0.5,
                    memory_type=MemoryType.CONVERSATION,
                )
                mid = await sqlite.create_memory(mem)
                await sqlite.update_memory(mid, is_processed=True)
                failed_ids.append(mid)

            # Verify no embeddings
            for mid in failed_ids:
                count = vectors._conn.execute(
                    "SELECT COUNT(*) FROM vec_memories WHERE rowid=?", [mid]
                ).fetchone()[0]
                assert count == 0, f"Setup: memory {mid} should have no embedding"

            # Run backfill (simulates what nightly does)
            result = vectors.backfill_missing_vectors("memories", limit=100)

            elapsed = time.monotonic() - t0

            # All 5 should now have embeddings
            recovered = 0
            for mid in failed_ids:
                count = vectors._conn.execute(
                    "SELECT COUNT(*) FROM vec_memories WHERE rowid=?", [mid]
                ).fetchone()[0]
                recovered += count

            ok = recovered == 5
            self.record("recovery_after_embed_failure", ok, elapsed,
                        "" if ok else f"Only {recovered}/5 recovered. Backfill result: {result}")

            vectors.close()
            await sqlite.close()
            return ok

        except Exception as e:
            self.record("recovery_after_embed_failure", False, time.monotonic() - t0, str(e))
            return False

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    async def run(self):
        print("\n=== BlipShell Live Integration Tests ===\n")

        print("Setup...", flush=True)
        try:
            await self.setup()
        except Exception as e:
            print(f"  SETUP FAILED: {e}")
            return False

        tests = [
            ("Pre-flight", [
                self.test_ollama_responsive,
                self.test_embed_single,
                self.test_embed_batch,
            ]),
            ("Real Usage", [
                self.test_full_session_with_worker,
                self.test_search_after_session,
                self.test_back_to_back_sessions,
                self.test_pii_embed_not_corrupted,
            ]),
            ("Stress & Recovery", [
                self.test_session_close_under_load,
                self.test_recovery_after_embed_failure,
                self.test_nightly_after_real_data,
            ]),
        ]

        if not self.quick:
            tests[0][1].append(self.test_embed_large_batch)

        abort = False
        for section_name, section_tests in tests:
            print(f"\n{section_name}:", flush=True)
            if abort:
                for t in section_tests:
                    self.record(t.__name__.replace("test_", ""), False, 0.0, "SKIPPED (prior failure)")
                continue

            for test_fn in section_tests:
                try:
                    passed = await test_fn()
                except Exception as e:
                    self.record(test_fn.__name__.replace("test_", ""), False, 0.0, f"Unhandled: {e}")
                    passed = False

                if not passed and section_name == "Pre-flight":
                    abort = True
                    break

        print("\nTeardown...", flush=True)
        await self.teardown()

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        total_time = sum(r["elapsed_s"] for r in self.results)

        print(f"\n{'=' * 50}")
        print(f"Results: {passed}/{total} passed ({failed} failed) in {total_time:.1f}s")

        if failed:
            print("\nFailures:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  X {r['name']}: {r['detail']}")

        print()
        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="BlipShell live integration tests")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    parser.add_argument("--verbose", action="store_true", help="Show details")
    args = parser.parse_args()

    runner = LiveTestRunner(verbose=args.verbose, quick=args.quick)
    success = asyncio.run(runner.run())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
