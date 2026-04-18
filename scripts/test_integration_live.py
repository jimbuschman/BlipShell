"""Live integration test — runs against real Ollama on the Ollama PC.

Tests the actual end-to-end pipeline with real models, real embeddings,
real search. Uses a temp database so production data is never touched.

This test catches the class of bugs that mock tests miss:
- Ollama crashes under load
- Embedding calls hang or timeout
- Session close fails with real LLM calls
- Search can't find memories with real embeddings
- PII sanitization corrupts real summaries
- Nightly jobs fail against real data

Usage:
  python scripts/test_integration_live.py              # full test
  python scripts/test_integration_live.py --quick      # skip slow tests
  python scripts/test_integration_live.py --verbose     # show details

Requires: Ollama running locally with nomic-embed-text available.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class LiveTestRunner:
    """Runs live integration tests against real Ollama."""

    def __init__(self, verbose: bool = False, quick: bool = False):
        self.verbose = verbose
        self.quick = quick
        self.results: list[dict] = []
        self.temp_db: str = ""
        self.sqlite = None
        self.vectors = None
        self.router = None
        self.config = None
        self.config_manager = None

    def log(self, msg: str):
        print(f"  {msg}", flush=True)

    def detail(self, msg: str):
        if self.verbose:
            print(f"    {msg}", flush=True)

    async def setup(self):
        """Initialize all components with a temp database."""
        from blipshell.core.config import ConfigManager
        from blipshell.llm.endpoints import EndpointManager
        from blipshell.llm.router import LLMRouter
        from blipshell.memory.sqlite_store import SQLiteStore
        from blipshell.memory.vector_store import VectorStore
        from blipshell.models.config import get_ollama_url

        # Use real config but temp database
        self.config_manager = ConfigManager(config_path=None)
        self.config = self.config_manager.load()

        fd, self.temp_db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.detail(f"Temp DB: {self.temp_db}")

        self.sqlite = SQLiteStore(self.temp_db)
        await self.sqlite.initialize()

        self.vectors = VectorStore(
            db_path=self.temp_db,
            embedding_model=self.config.models.embedding,
            ollama_url=get_ollama_url(self.config.endpoints),
            embedding_dim=self.config.database.embedding_dimensions,
        )
        self.vectors.initialize()

        endpoint_manager = EndpointManager(self.config.endpoints, self.config.llm)
        self.router = LLMRouter(
            self.config.models, endpoint_manager,
            pii_enabled=self.config.pii.enabled,
        )

    async def teardown(self):
        """Clean up temp database."""
        if self.sqlite:
            await self.sqlite.close()
        if self.vectors:
            self.vectors.close()
        if self.temp_db and os.path.exists(self.temp_db):
            try:
                os.unlink(self.temp_db)
            except OSError:
                pass

    def record(self, name: str, passed: bool, elapsed: float, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "name": name,
            "passed": passed,
            "elapsed_s": round(elapsed, 2),
            "detail": detail,
        })
        icon = "+" if passed else "X"
        print(f"  [{icon}] {name} ({elapsed:.1f}s){f' — {detail}' if detail and not passed else ''}", flush=True)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    async def test_ollama_responsive(self):
        """Ollama is running and the embedding model is available."""
        t0 = time.monotonic()
        try:
            import ollama
            client = ollama.Client(host="http://localhost:11434")
            models = client.list()
            model_names = [m.model for m in models.models]
            embed_model = self.config.models.embedding
            found = any(embed_model in name for name in model_names)
            elapsed = time.monotonic() - t0
            if not found:
                self.record("ollama_responsive", False, elapsed,
                            f"Embedding model '{embed_model}' not found. Available: {model_names[:5]}")
                return False
            self.record("ollama_responsive", True, elapsed)
            return True
        except Exception as e:
            self.record("ollama_responsive", False, time.monotonic() - t0, str(e))
            return False

    async def test_embed_single(self):
        """Single embedding call completes within timeout."""
        t0 = time.monotonic()
        try:
            vec = self.vectors._embed("This is a test sentence for embedding.")
            elapsed = time.monotonic() - t0
            ok = len(vec) == self.config.database.embedding_dimensions
            self.record("embed_single", ok, elapsed,
                        "" if ok else f"Expected dim {self.config.database.embedding_dimensions}, got {len(vec)}")
            return ok
        except Exception as e:
            self.record("embed_single", False, time.monotonic() - t0, str(e))
            return False

    async def test_embed_batch(self):
        """Batch embedding (32 items) completes without hanging."""
        t0 = time.monotonic()
        try:
            texts = [f"Test sentence number {i} about various topics." for i in range(32)]
            vecs = self.vectors._embed_batch(texts)
            elapsed = time.monotonic() - t0
            ok = len(vecs) == 32
            self.record("embed_batch_32", ok, elapsed,
                        "" if ok else f"Expected 32 embeddings, got {len(vecs)}")
            return ok
        except Exception as e:
            self.record("embed_batch_32", False, time.monotonic() - t0, str(e))
            return False

    async def test_embed_large_batch(self):
        """Large batch (100 items) completes via chunking."""
        t0 = time.monotonic()
        try:
            texts = [f"Test sentence number {i} about topic {i % 10}." for i in range(100)]
            vecs = self.vectors._embed_batch(texts)
            elapsed = time.monotonic() - t0
            ok = len(vecs) == 100
            self.record("embed_batch_100", ok, elapsed,
                        "" if ok else f"Expected 100 embeddings, got {len(vecs)}")
            return ok
        except Exception as e:
            self.record("embed_batch_100", False, time.monotonic() - t0, str(e))
            return False

    async def test_summarize(self):
        """Summarization LLM call works."""
        t0 = time.monotonic()
        try:
            from blipshell.llm.router import TaskType
            summary = await self.router.generate(
                TaskType.SUMMARIZATION,
                "The user discussed their plans to learn Python for data science. They mentioned wanting to build machine learning models.",
            )
            elapsed = time.monotonic() - t0
            ok = len(summary) > 10 and summary.strip().upper() != "SKIP"
            self.record("summarize", ok, elapsed,
                        "" if ok else f"Bad summary: {summary[:100]}")
            return ok
        except Exception as e:
            self.record("summarize", False, time.monotonic() - t0, str(e))
            return False

    async def test_rank_importance(self):
        """Rank+importance scoring LLM call returns parseable results."""
        t0 = time.monotonic()
        try:
            from blipshell.llm.router import TaskType
            result = await self.router.generate(
                TaskType.RANKING_IMPORTANCE,
                "User discussed Python performance tuning and optimization strategies.",
            )
            elapsed = time.monotonic() - t0
            parts = result.strip().split()
            ok = len(parts) >= 2
            if ok:
                try:
                    rank = int(parts[0])
                    importance = float(parts[1])
                    ok = 1 <= rank <= 5 and 0.0 <= importance <= 1.0
                except (ValueError, IndexError):
                    ok = False
            self.record("rank_importance", ok, elapsed,
                        "" if ok else f"Unparseable: {result[:100]}")
            return ok
        except Exception as e:
            self.record("rank_importance", False, time.monotonic() - t0, str(e))
            return False

    async def test_process_message(self):
        """Full process_message pipeline: summarize → embed → score."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.processor import MemoryProcessor

            processor = MemoryProcessor(
                self.sqlite, self.vectors, self.router,
                config=self.config.memory,
            )
            sid = await self.sqlite.create_session(title="Live test session")

            mem_id = await processor.process_message(
                text="Kortney and I talked about waiting until Monday to reconnect. She needs some space after everything that happened.",
                role="user",
                session_id=sid,
            )
            elapsed = time.monotonic() - t0

            if not mem_id:
                self.record("process_message", False, elapsed, "No memory ID returned")
                return False

            # Verify memory has summary
            memory = await self.sqlite.get_memory(mem_id)
            has_summary = memory and memory.summary and len(memory.summary) > 5

            # Verify embedding exists
            vec_count = self.vectors._conn.execute(
                "SELECT COUNT(*) FROM vec_memories WHERE rowid=?", [mem_id]
            ).fetchone()[0]
            has_embed = vec_count == 1

            ok = has_summary and has_embed
            detail = ""
            if not has_summary:
                detail += "Missing summary. "
            if not has_embed:
                detail += "Missing embedding. "
            self.record("process_message", ok, elapsed, detail)
            return ok
        except Exception as e:
            self.record("process_message", False, time.monotonic() - t0, str(e))
            return False

    async def test_embed_uses_content_not_summary(self):
        """Embedding is generated from raw content, not PII-sanitized summary."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.processor import MemoryProcessor

            processor = MemoryProcessor(
                self.sqlite, self.vectors, self.router,
                config=self.config.memory,
            )
            sid = await self.sqlite.create_session(title="PII embed test")

            mem_id = await processor.process_message(
                text="Kortney texted me at 3pm on Monday about meeting at the coffee shop on Oak Street.",
                role="user",
                session_id=sid,
            )
            elapsed = time.monotonic() - t0

            if not mem_id:
                self.record("embed_uses_content", False, elapsed, "No memory ID")
                return False

            # Search for "Kortney" — should find it via vector similarity
            # because embedding was generated from content (has "Kortney"),
            # not from summary (which might have [PERSON])
            results = self.vectors.search_memories("Kortney Monday coffee", n_results=5)
            found = any(r["id"] == mem_id for r in results)

            # Also check if summary got PII-sanitized
            memory = await self.sqlite.get_memory(mem_id)
            pii_in_summary = "[PERSON]" in (memory.summary or "") or "[PII]" in (memory.summary or "")

            ok = found
            detail = ""
            if pii_in_summary:
                detail += "Summary was PII-sanitized (expected if using cloud). "
            if not found:
                detail += f"Vector search for 'Kortney Monday coffee' did not find memory {mem_id}. "
                # Show what was found
                if results:
                    detail += f"Found IDs: {[r['id'] for r in results]}. "
                else:
                    detail += "No vector results at all. "

            self.record("embed_uses_content", ok, elapsed, detail)
            return ok
        except Exception as e:
            self.record("embed_uses_content", False, time.monotonic() - t0, str(e))
            return False

    async def test_search_finds_keyword(self):
        """Full search pipeline finds memories by keyword."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.search import MemorySearch
            from blipshell.memory.processor import MemoryProcessor

            processor = MemoryProcessor(
                self.sqlite, self.vectors, self.router,
                config=self.config.memory,
            )
            sid = await self.sqlite.create_session(title="Search test session")

            # Insert a memory with a unique keyword
            await processor.process_message(
                text="Xylophone lessons with Bartholomew start next Thursday at the community center. He's been teaching for years.",
                role="user",
                session_id=sid,
            )

            # Search from a different session
            sid2 = await self.sqlite.create_session(title="Search query session")
            search = MemorySearch(
                self.sqlite, self.vectors, self.router,
                config=self.config.memory,
            )
            results = await search.search(
                query="Xylophone Bartholomew Thursday",
                current_session_id=sid2,
                n_results=10,
            )
            elapsed = time.monotonic() - t0

            found = len(results) > 0
            detail = ""
            if not found:
                # Debug: check FTS directly
                fts = await self.sqlite.search_fts("Xylophone", limit=5)
                detail = f"0 results. FTS for 'Xylophone': {len(fts)} hits."
            else:
                detail = f"{len(results)} results, top score={results[0].boosted_score:.3f}"
                self.detail(detail)

            self.record("search_finds_keyword", found, elapsed, detail if not found else "")
            return found
        except Exception as e:
            self.record("search_finds_keyword", False, time.monotonic() - t0, str(e))
            return False

    async def test_session_close(self):
        """Session close completes: message_count set, title generated."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.processor import MemoryProcessor
            from blipshell.memory.manager import MemoryManager
            from blipshell.session.manager import SessionManager
            from blipshell.models.session import MessageRole

            processor = MemoryProcessor(
                self.sqlite, self.vectors, self.router,
                config=self.config.memory,
            )
            mm = MemoryManager(self.config.memory)
            sm = SessionManager(self.sqlite, mm, processor, self.router)

            sid = await sm.start_session()
            messages = [
                (MessageRole.USER, "I've been thinking about learning to play guitar."),
                (MessageRole.ASSISTANT, "That's a great idea! What kind of music are you interested in?"),
                (MessageRole.USER, "Mostly classic rock. I grew up listening to Led Zeppelin and Pink Floyd."),
                (MessageRole.ASSISTANT, "Classic rock is a great foundation. I'd suggest starting with basic open chords."),
                (MessageRole.USER, "Should I get an acoustic or electric to start?"),
                (MessageRole.ASSISTANT, "Acoustic is usually recommended for beginners — it builds finger strength."),
            ]
            for role, content in messages:
                sm.add_message(role, content)

            await sm.flush_pending_persists()
            await sm.end_session()
            elapsed = time.monotonic() - t0

            session = await self.sqlite.get_session(sid)
            checks = []
            if session.message_count == 0:
                checks.append(f"message_count=0 (expected {len(messages)})")
            if session.title == "New Session":
                checks.append("title still 'New Session'")
            if not session.summary:
                checks.append("no summary generated")

            # Check memories exist
            memories = await self.sqlite.get_memories_by_session(sid)
            if len(memories) == 0:
                checks.append("no memories persisted")

            ok = len(checks) == 0
            self.record("session_close", ok, elapsed, "; ".join(checks) if checks else "")
            return ok
        except Exception as e:
            self.record("session_close", False, time.monotonic() - t0, str(e))
            return False

    async def test_session_close_with_timeout(self):
        """Session close doesn't hang even if LLM is slow."""
        t0 = time.monotonic()
        try:
            from blipshell.memory.processor import MemoryProcessor
            from blipshell.memory.manager import MemoryManager
            from blipshell.session.manager import SessionManager
            from blipshell.models.session import MessageRole

            processor = MemoryProcessor(
                self.sqlite, self.vectors, self.router,
                config=self.config.memory,
            )
            mm = MemoryManager(self.config.memory)
            sm = SessionManager(self.sqlite, mm, processor, self.router)

            sid = await sm.start_session()
            sm.add_message(MessageRole.USER, "Quick test message about timeout handling.")
            sm.add_message(MessageRole.ASSISTANT, "This is a short session to test close timing.")

            await sm.flush_pending_persists()

            # end_session should complete within 60s even if LLM is slow
            try:
                await asyncio.wait_for(sm.end_session(), timeout=60.0)
                elapsed = time.monotonic() - t0
                ok = True
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - t0
                ok = False
                self.record("session_close_timeout", False, elapsed, "end_session took >60s")
                return False

            # Even if summary failed, message_count should be set
            session = await self.sqlite.get_session(sid)
            has_count = session.message_count > 0
            self.record("session_close_timeout", has_count, elapsed,
                        "" if has_count else f"message_count={session.message_count}")
            return has_count
        except Exception as e:
            self.record("session_close_timeout", False, time.monotonic() - t0, str(e))
            return False

    async def test_nightly_backfill(self):
        """Nightly backfill_vectors finds and fixes missing embeddings."""
        t0 = time.monotonic()
        try:
            from blipshell.models.memory import Memory, MemoryType

            sid = await self.sqlite.create_session(title="Backfill test")

            # Create a memory with summary but no embedding
            mem = Memory(
                session_id=sid, role="user",
                content="Unique test content for backfill verification.",
                summary="Unique backfill test summary.",
                timestamp=datetime.now(timezone.utc),
                rank=3, importance=0.5,
                memory_type=MemoryType.CONVERSATION,
            )
            mid = await self.sqlite.create_memory(mem)
            await self.sqlite.update_memory(mid, is_processed=True)

            # Verify no embedding yet
            before = self.vectors._conn.execute(
                "SELECT COUNT(*) FROM vec_memories WHERE rowid=?", [mid]
            ).fetchone()[0]
            assert before == 0, "Setup: should have no embedding"

            # Run backfill
            result = self.vectors.backfill_missing_vectors("memories", limit=100)
            elapsed = time.monotonic() - t0

            after = self.vectors._conn.execute(
                "SELECT COUNT(*) FROM vec_memories WHERE rowid=?", [mid]
            ).fetchone()[0]

            ok = after == 1
            self.record("nightly_backfill", ok, elapsed,
                        "" if ok else f"Backfill result: {result}")
            return ok
        except Exception as e:
            self.record("nightly_backfill", False, time.monotonic() - t0, str(e))
            return False

    async def test_orphan_vector_cleanup(self):
        """Orphan vector cleanup removes vectors for archived memories."""
        t0 = time.monotonic()
        try:
            from blipshell.models.memory import Memory, MemoryType

            sid = await self.sqlite.create_session(title="Orphan test")
            mem = Memory(
                session_id=sid, role="user",
                content="This memory will be archived to test orphan cleanup.",
                summary="Orphan cleanup test.",
                timestamp=datetime.now(timezone.utc),
                rank=1, importance=0.1,
                memory_type=MemoryType.CONVERSATION,
            )
            mid = await self.sqlite.create_memory(mem)
            await self.sqlite.update_memory(mid, is_processed=True)

            # Embed it
            self.vectors.add_memory(mid, mem.content)

            # Archive it (simulating prune)
            await self.sqlite.update_memory(mid, is_archived=True)

            # Cleanup
            result = self.vectors.cleanup_orphan_vectors()
            elapsed = time.monotonic() - t0

            after = self.vectors._conn.execute(
                "SELECT COUNT(*) FROM vec_memories WHERE rowid=?", [mid]
            ).fetchone()[0]

            ok = after == 0
            self.record("orphan_cleanup", ok, elapsed,
                        "" if ok else f"Vector still exists after cleanup: {result}")
            return ok
        except Exception as e:
            self.record("orphan_cleanup", False, time.monotonic() - t0, str(e))
            return False

    async def run(self):
        """Run all tests."""
        print("\n=== BlipShell Live Integration Tests ===\n")

        # Pre-flight
        print("Setup...", flush=True)
        try:
            await self.setup()
        except Exception as e:
            print(f"  SETUP FAILED: {e}")
            return False

        # Ordered tests — each depends on prior passing
        tests = [
            ("Pre-flight", [
                self.test_ollama_responsive,
                self.test_embed_single,
                self.test_embed_batch,
            ]),
            ("LLM Pipeline", [
                self.test_summarize,
                self.test_rank_importance,
                self.test_process_message,
            ]),
            ("Search & PII", [
                self.test_embed_uses_content_not_summary,
                self.test_search_finds_keyword,
            ]),
            ("Session Lifecycle", [
                self.test_session_close,
                self.test_session_close_with_timeout,
            ]),
            ("Maintenance", [
                self.test_nightly_backfill,
                self.test_orphan_vector_cleanup,
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

                # Abort remaining sections if pre-flight fails
                if not passed and section_name == "Pre-flight":
                    abort = True
                    break

        # Teardown
        print("\nTeardown...", flush=True)
        await self.teardown()

        # Summary
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
