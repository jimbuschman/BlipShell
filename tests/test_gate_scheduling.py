"""Scheduling under load: a foreground chat turn owns the local model.

Live lifecycle test 2026-09-16 (docs/ACCEPTANCE_2026_09_16.md, "Remaining live
findings"): background entity extraction ran while a chat turn was in flight;
the turn's search embeddings timed out, retrieval fell back to keywords, and
the newly queued conversation messages waited behind maintenance. Two gaps:
search embeddings were not gated at all (`search_memories` was documented as
"NOT gated"), and the worker judged the system idle from its own queue alone.

The fix (07b8930 + this file's follow-ups): every embedding call takes the
gate; a chat turn holds `interactive_turn()`, during which NEW background
acquisitions park (a running call is never interrupted); priority is a
contextvar carried across `asyncio.to_thread`; the worker's idle branch also
checks the turn.

These drive the REAL gate, VectorStore, MemorySearch and decorators with real
threads and event loops. Only the embedding client is faked.
"""

import asyncio
import re
import threading
import time
from pathlib import Path

import pytest

from blipshell.llm.ollama_gate import (
    BACKGROUND,
    INTERACTIVE,
    GateTimeout,
    OllamaGate,
    background_model_work,
    get_gate,
    interactive_model_work,
)
from blipshell.memory.search import MemorySearch
from blipshell.memory.vector_store import VectorStore

REPO = Path(__file__).resolve().parents[1]


def _poll(condition, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


async def _apoll(condition, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        await asyncio.sleep(0.005)
    return condition()


# --- The gate: an interactive turn parks NEW background work ----------------

class TestInteractiveTurn:
    async def test_background_parks_during_a_turn_even_when_the_gate_is_idle(self):
        """The core case: nothing is running, a chat turn is in progress, and
        the worker wants the model. It must wait for the turn, not slip in
        ahead of the turn's own (not yet issued) embedding and generation."""
        gate = OllamaGate()
        with gate.interactive_turn():
            assert gate.interactive_active
            bg = asyncio.create_task(gate.async_acquire(BACKGROUND))
            assert await _apoll(lambda: gate.waiter_count == 1)
            assert not bg.done()
            assert not gate.is_active           # parked: nobody holds the gate

            # The turn's own calls are unaffected.
            assert await asyncio.wait_for(gate.async_acquire(INTERACTIVE), 1.0)
            gate.release()
            await asyncio.sleep(0.05)
            assert not bg.done(), "a release inside the turn woke background work"

        # Turn over: the parked waiter is woken without any further release().
        assert await asyncio.wait_for(bg, 1.0)
        assert gate.is_active
        gate.release()
        assert not gate.is_active
        assert not gate.interactive_active

    async def test_a_running_background_call_is_not_interrupted(self):
        """No preemption. The worker's in-flight call finishes; the turn's
        call goes next, ahead of any further queued background work, which
        stays parked until the turn ends."""
        gate = OllamaGate()
        await gate.async_acquire(BACKGROUND)    # worker mid-generation
        order: list[str] = []

        async def take(priority, label):
            await gate.async_acquire(priority)
            order.append(label)
            gate.release()

        with gate.interactive_turn():
            bg2 = asyncio.create_task(take(BACKGROUND, "bg"))
            assert await _apoll(lambda: gate.waiter_count == 1)
            fg = asyncio.create_task(take(INTERACTIVE, "chat"))
            assert await _apoll(lambda: gate.waiter_count == 2)

            gate.release()                       # the worker's call ends
            await asyncio.wait_for(fg, 1.0)
            assert order == ["chat"]
            await asyncio.sleep(0.05)
            assert not bg2.done()

        await asyncio.wait_for(bg2, 1.0)
        assert order == ["chat", "bg"]
        assert not gate.is_active

    def test_sync_background_acquire_parks_during_a_turn(self):
        """The VectorStore path is synchronous (`gate.gate()` on a pool
        thread): same rule."""
        gate = OllamaGate()
        acquired = threading.Event()

        def contender():
            with gate.gate(BACKGROUND):
                acquired.set()

        with gate.interactive_turn():
            t = threading.Thread(target=contender)
            t.start()
            assert _poll(lambda: gate.waiter_count == 1)
            time.sleep(0.1)
            assert not acquired.is_set()
        t.join(2.0)
        assert acquired.is_set()
        assert not gate.is_active

    async def test_nested_turns_release_background_only_when_all_end(self):
        gate = OllamaGate()
        with gate.interactive_turn():
            with gate.interactive_turn():
                bg = asyncio.create_task(gate.async_acquire(BACKGROUND))
                assert await _apoll(lambda: gate.waiter_count == 1)
            await asyncio.sleep(0.05)
            assert not bg.done(), "inner turn ending released background work"
        assert await asyncio.wait_for(bg, 1.0)
        gate.release()
        assert not gate.is_active

    async def test_a_turn_that_raises_still_releases_background(self):
        gate = OllamaGate()
        bg = None
        with pytest.raises(ValueError):
            with gate.interactive_turn():
                bg = asyncio.create_task(gate.async_acquire(BACKGROUND))
                assert await _apoll(lambda: gate.waiter_count == 1)
                raise ValueError("turn failed")
        assert await asyncio.wait_for(bg, 1.0)
        gate.release()
        assert not gate.is_active

    async def test_parked_waiter_can_time_out_and_withdraw(self):
        gate = OllamaGate()
        with gate.interactive_turn():
            with pytest.raises(GateTimeout):
                await gate.async_acquire(BACKGROUND, timeout=0.05)
            assert gate.waiter_count == 0
        assert not gate.is_active
        assert await gate.async_acquire(BACKGROUND, timeout=0.5)
        gate.release()

    async def test_parked_waiter_can_be_cancelled(self):
        """Worker shutdown while parked (Ctrl+C mid-turn) must not wedge."""
        gate = OllamaGate()
        with gate.interactive_turn():
            bg = asyncio.create_task(gate.async_acquire(BACKGROUND))
            assert await _apoll(lambda: gate.waiter_count == 1)
            bg.cancel()
            with pytest.raises(asyncio.CancelledError):
                await bg
            assert gate.waiter_count == 0
        assert not gate.is_active
        assert await asyncio.wait_for(gate.async_acquire(BACKGROUND), 1.0)
        gate.release()


# --- Sync reentrancy: add_memory (gated) calls _embed (gated) ----------------

class TestSyncReentrancy:
    def test_nested_sync_gate_on_one_thread_does_not_deadlock(self):
        gate = OllamaGate()
        with gate.gate(BACKGROUND):
            assert gate.is_active
            with gate.gate(BACKGROUND):
                assert gate.is_active
            assert gate.is_active, "the inner exit released the outer hold"
        assert not gate.is_active

    def test_reentrancy_is_per_thread(self):
        """Another thread does not ride the holder's depth: it waits."""
        gate = OllamaGate()
        acquired = threading.Event()

        def contender():
            with gate.gate(BACKGROUND):
                acquired.set()

        with gate.gate(INTERACTIVE):
            t = threading.Thread(target=contender)
            t.start()
            assert _poll(lambda: gate.waiter_count == 1)
            assert not acquired.is_set()
        t.join(2.0)
        assert acquired.is_set()
        assert not gate.is_active

    def test_depth_is_not_left_set_by_a_timed_out_acquire(self):
        """If a GateTimeout left depth at 1, the thread's NEXT gate() would
        yield without acquiring: an unserialised Ollama call."""
        gate = OllamaGate()
        holder_done = threading.Event()
        release_now = threading.Event()

        def holder():
            with gate.gate(BACKGROUND):
                release_now.wait(2.0)
            holder_done.set()

        t = threading.Thread(target=holder)
        t.start()
        assert _poll(lambda: gate.is_active)
        with pytest.raises(GateTimeout):
            with gate.gate(INTERACTIVE, timeout=0.05):
                pass
        release_now.set()
        t.join(2.0)
        assert holder_done.is_set()
        with gate.gate(INTERACTIVE):
            assert gate.is_active, "gate() yielded without acquiring"
        assert not gate.is_active


# --- Priority is a contextvar and must survive the thread hop ---------------

async def _stub():
    return None


class TestPriorityContext:
    async def test_background_context_survives_asyncio_to_thread(self):
        gate = OllamaGate()

        @background_model_work
        async def job():
            assert gate.infer_priority() == BACKGROUND
            return await asyncio.to_thread(gate.infer_priority)

        assert await job() == BACKGROUND
        assert gate.infer_priority() == INTERACTIVE   # main thread, no context

    async def test_run_in_executor_drops_the_context(self):
        """Why every model-touching thread hop uses asyncio.to_thread: the
        plain executor does not copy contextvars, so a background caller's
        embedding would reach the gate as INTERACTIVE and skip the pause."""
        gate = OllamaGate()

        @background_model_work
        async def job():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, gate.infer_priority)

        assert await job() == INTERACTIVE

    async def test_interactive_model_work_owns_the_turn_for_its_duration(self):
        gate = get_gate()
        seen = {}

        @interactive_model_work
        async def turn():
            seen["active"] = gate.interactive_active
            seen["priority"] = gate.infer_priority()
            seen["in_thread"] = await asyncio.to_thread(gate.infer_priority)

        assert not gate.interactive_active
        await turn()
        assert seen == {"active": True, "priority": INTERACTIVE, "in_thread": INTERACTIVE}
        assert not gate.interactive_active

    async def test_interactive_model_work_releases_the_turn_on_error(self):
        gate = get_gate()

        @interactive_model_work
        async def turn():
            raise RuntimeError("model down")

        with pytest.raises(RuntimeError):
            await turn()
        assert not gate.interactive_active

    def test_the_production_entry_points_are_wrapped(self):
        """The decorators are only worth anything on the real entry points.
        Every application of a decorator shares ONE code object, so identity
        on __code__ proves which decorator wraps the method."""
        from blipshell.core.agent import Agent
        from blipshell.core.agent_chat import ChatMixin
        from blipshell.core.nightly import NightlyRunner
        from blipshell.memory.worker import MemoryWorker

        interactive_code = interactive_model_work(_stub).__code__
        background_code = background_model_work(_stub).__code__

        assert ChatMixin.chat.__code__ is interactive_code
        assert MemoryWorker._run.__code__ is background_code
        assert NightlyRunner.run.__code__ is background_code
        assert Agent._reflection_loop.__code__ is background_code
        assert Agent._reflect_on_return.__code__ is background_code

    def test_no_model_touching_thread_hop_uses_run_in_executor(self):
        """Source pin for the propagation rule. `run_in_executor` is fine for
        DB reads, queue polls and backups; anything reaching the vector store
        (an embedding) must be `asyncio.to_thread`."""
        # VectorStore reads that never call Ollama (stored vectors only) may
        # keep the plain executor; the rule is about EMBEDDING, not the class.
        non_embedding = ("find_neighbors", "get_embeddings_by_ids")
        offenders = []
        for rel in ("blipshell/memory/search.py", "blipshell/core/nightly.py",
                    "blipshell/core/agent.py", "blipshell/core/agent_chat.py",
                    "blipshell/memory/processor.py", "blipshell/memory/worker.py",
                    "blipshell/memory/consolidation.py"):
            src = (REPO / rel).read_text(encoding="utf-8")
            for m in re.finditer(r"run_in_executor\((.{0,240})", src, re.S):
                hop = m.group(1)
                if any(name in hop for name in non_embedding):
                    continue
                if "vectors" in hop or "embed" in hop:
                    offenders.append(f"{rel}: {m.group(0)[:80]!r}")
        assert not offenders, offenders


# --- VectorStore: query embeddings take the gate ----------------------------

class FakeEmbedder:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list = []

    def embed(self, model: str, input):  # noqa: A002 - ollama's kwarg name
        items = [input] if isinstance(input, str) else list(input)
        self.calls.append(items)
        return {"embeddings": [[float(len(t))] * self.dim for t in items]}


@pytest.fixture
def vectors(tmp_path):
    v = VectorStore(db_path=str(tmp_path / "v.db"), embedding_model="fake",
                    ollama_url="http://localhost:1", embedding_dim=8)
    v.initialize()
    v._ollama_client = FakeEmbedder(dim=8)
    yield v
    v.close()


class TestSearchEmbeddingsAreGated:
    def test_query_embedding_waits_for_a_running_background_call(self, vectors):
        """`search_memories` used to be documented "NOT gated"; its embedding
        ran alongside a background generation on the same GPU and timed out.
        Now it queues behind the running call like every other Ollama call."""
        gate = get_gate()
        gate.acquire(BACKGROUND)                 # the worker, mid-generation
        released = False
        done = threading.Event()

        def searcher():
            vectors.search_memories(query="what did we decide", n_results=3)
            done.set()

        t = threading.Thread(target=searcher)
        t.start()
        try:
            assert _poll(lambda: gate.waiter_count == 1), "search did not take the gate"
            time.sleep(0.05)
            assert vectors._ollama_client.calls == [], (
                "the query embedding ran while the gate was held"
            )
            gate.release()
            released = True
            assert done.wait(2.0)
            assert vectors._ollama_client.calls == [["what did we decide"]]
        finally:
            if not released:
                gate.release()
            t.join(2.0)
        assert not gate.is_active

    def test_write_path_embeds_under_one_hold(self, vectors):
        """add_memory (gated) -> _embed (gated): the production reentrant
        pair. Completes, embeds once, and leaves the gate open."""
        gate = get_gate()
        vectors.add_memory(1, "hello world", {"session_id": "1"})
        assert vectors._ollama_client.calls == [["hello world"]]
        assert not gate.is_active


# --- MemorySearch: the caller's priority reaches the embedding ---------------

class RecordingVectors:
    """Stands in for VectorStore; records the gate priority each call sees."""

    def __init__(self):
        self.seen: list[int] = []

    def _record(self, *_a, **_k):
        self.seen.append(get_gate().infer_priority())
        return []

    def __getattr__(self, name):
        if name.startswith("search_") or name == "embed_text":
            return self._record
        raise AttributeError(name)


class TestSearchPropagatesCallerPriority:
    async def test_pool_searches_carry_background_priority_from_a_background_caller(self):
        rec = RecordingVectors()
        ms = MemorySearch(sqlite=None, vectors=rec, router=None)

        await ms.search_core_memories("q")
        await ms.search_lessons("q")
        assert rec.seen == [INTERACTIVE] * 3     # chat: main thread, no context

        @background_model_work
        async def maintenance():
            await ms.search_core_memories("q")
            await ms.search_lessons("q")

        await maintenance()
        assert rec.seen[3:] == [BACKGROUND] * 3

    async def test_recall_search_carries_background_priority(self, sqlite_store,
                                                             canned_router, memory_config):
        """The full recall path (`search`), whose vector pass is the one the
        live test saw time out."""
        rec = RecordingVectors()
        ms = MemorySearch(sqlite_store, rec, canned_router, memory_config)

        await ms.search("what did we decide about the gate")
        assert rec.seen and set(rec.seen) == {INTERACTIVE}
        n = len(rec.seen)

        @background_model_work
        async def maintenance():
            await ms.search("what did we decide about the gate")

        await maintenance()
        assert rec.seen[n:] and set(rec.seen[n:]) == {BACKGROUND}
