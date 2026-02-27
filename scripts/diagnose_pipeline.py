"""Diagnose memory processing pipeline performance.

Runs test messages through the full pipeline with per-step timing:
- Noise check, summarize (LLM), SQLite insert, ChromaDB embed, dedup (LLM?), tag, rank+importance (LLM)
- Which endpoint/model handles each LLM step
- Whether dedup fires an extra LLM call
- Concurrent DB access test (reproduces "database is locked")
- Total per-message processing time and bottleneck identification

Usage:
    python scripts/diagnose_pipeline.py
    python scripts/diagnose_pipeline.py --messages 3
    python scripts/diagnose_pipeline.py --config path/to/config.yaml
    python scripts/diagnose_pipeline.py --concurrent-only   # just test DB locking
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.core.config import ConfigManager
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose")

# Keep routing visible
logging.getLogger("blipshell.llm.endpoints").setLevel(logging.DEBUG)
logging.getLogger("blipshell.llm.router").setLevel(logging.DEBUG)
logging.getLogger("blipshell.memory.processor").setLevel(logging.DEBUG)


TEST_MESSAGES = [
    ("user", "I've been working on improving the memory system. The main issue is that "
     "conversations aren't being stored reliably when the session closes."),
    ("assistant", "I can help with that. The session close process needs to drain the "
     "memory worker queue before running summary generation. Let me check the code."),
    ("user", "Also the entity extraction seems slow. Can we batch the LLM calls?"),
    ("assistant", "Good idea. Currently each message gets its own summarization and ranking "
     "call. We could batch 5 messages into one ranking call to reduce overhead."),
    ("user", "What about using Groq for the summarization? It should be faster than local."),
]


async def diagnose_routing(router: LLMRouter, endpoint_mgr: EndpointManager):
    """Show which endpoints handle which task types."""
    print("\n" + "=" * 70)
    print("ENDPOINT ROUTING MAP")
    print("=" * 70)

    task_types = [
        TaskType.SUMMARIZATION,
        TaskType.RANKING,
        TaskType.RANKING_IMPORTANCE,
        TaskType.IMPORTANCE,
        TaskType.REASONING,
        TaskType.TOOL_CALLING,
        TaskType.CODING,
    ]

    for tt in task_types:
        ep = await endpoint_mgr.get_endpoint_for_role(tt)
        if ep:
            model = ep.models.get(tt) or router.get_model(tt)
            rate_info = ""
            if ep.rate_limit_rpm:
                rate_info += f" rpm={ep.rate_limit_rpm}"
            if ep.rate_limit_rpd:
                rate_info += f" rpd={ep.rate_limit_rpd}"
            print(f"  {tt:25s} -> {ep.name:15s} model={model}{rate_info}")
        else:
            print(f"  {tt:25s} -> NO ENDPOINT AVAILABLE")

    print()
    print("All endpoints:")
    for ep in endpoint_mgr._endpoints:
        status = "OK" if ep.can_accept_request else "UNAVAILABLE"
        if ep._is_rate_limited():
            status = "RATE LIMITED"
        print(f"  {ep.name:15s} provider={ep.provider:8s} priority={ep.priority} "
              f"roles={ep.roles} status={status}")
    print()


async def diagnose_individual_calls(router: LLMRouter, endpoint_mgr: EndpointManager):
    """Time individual LLM calls for each task type used in the pipeline."""
    print("=" * 70)
    print("INDIVIDUAL LLM CALL TIMING")
    print("=" * 70)
    print("  (Testing each task type the pipeline uses)\n")

    test_cases = [
        (TaskType.SUMMARIZATION, "Summarize this conversation message:\n"
         "User discussed improving the session close process for memory persistence."),
        (TaskType.RANKING_IMPORTANCE, "Rate this memory:\n"
         "rank (1-5) importance (0.0-1.0) type (fact/event/preference/skill/conversation)\n"
         "Memory: User discussed improving the session close process for memory persistence."),
        (TaskType.REASONING, "Given the new memory and existing memories, decide: ADD/UPDATE/DELETE/NONE.\n"
         "New: User wants to improve memory persistence.\n"
         "Existing 1: User discussed session management improvements."),
    ]

    for tt, prompt in test_cases:
        ep = await endpoint_mgr.get_endpoint_for_role(tt)
        if not ep:
            print(f"  {tt:25s} -> NO ENDPOINT")
            continue

        model = ep.models.get(tt) or router.get_model(tt)
        print(f"  {tt:25s} -> {ep.name} ({model})")

        t0 = time.monotonic()
        try:
            result = await router.generate(tt, prompt)
            elapsed = time.monotonic() - t0
            preview = result[:80].replace("\n", " ") if result else "(empty)"
            print(f"    Time: {elapsed:.2f}s")
            print(f"    Response: {preview}")
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"    FAILED after {elapsed:.2f}s: {e}")
        print()

    # Test embedding separately
    print(f"  embedding")
    t0 = time.monotonic()
    try:
        from blipshell.memory.chroma_store import ChromaStore
        chroma = ChromaStore()
        chroma._collection.add(
            ids=["diag_test_001"],
            documents=["Test embedding for diagnostic purposes"],
            metadatas=[{"session_id": "0", "role": "user"}],
        )
        elapsed = time.monotonic() - t0
        print(f"    Time: {elapsed:.3f}s")
        chroma._collection.delete(ids=["diag_test_001"])
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"    FAILED after {elapsed:.2f}s: {e}")
    print()


async def diagnose_full_pipeline(config, num_messages: int):
    """Run test messages through the full memory processing pipeline with per-step instrumentation."""
    print("=" * 70)
    print(f"FULL PIPELINE TEST ({num_messages} messages)")
    print("=" * 70)
    print("  Pipeline: noise_check -> summarize(LLM) -> sqlite_insert -> embed")
    print("            -> dedup(chroma_search + LLM?) -> tag(regex) -> rank+importance(LLM)")
    print()

    sqlite = SQLiteStore(config.database.path)
    await sqlite.initialize()

    endpoint_mgr = EndpointManager(config.endpoints, config.llm)
    router = LLMRouter(config.models, endpoint_mgr)

    try:
        from blipshell.memory.chroma_store import ChromaStore
        chroma = ChromaStore()
    except Exception as e:
        print(f"  ChromaDB unavailable: {e}")
        await sqlite.close()
        return

    from blipshell.memory.processor import MemoryProcessor
    processor = MemoryProcessor(
        sqlite, chroma, router,
        config=config.memory,
        max_tags=config.tagging.max_tags,
    )

    # Patch router.generate to capture per-call routing + timing
    original_generate = router.generate
    call_log: list[dict] = []

    async def patched_generate(task_type, prompt, **kwargs):
        ep = await endpoint_mgr.get_endpoint_for_role(task_type)
        ep_name = ep.name if ep else "none"
        model = (ep.models.get(task_type) or router.get_model(task_type)) if ep else "none"
        t0 = time.monotonic()
        try:
            result = await original_generate(task_type, prompt, **kwargs)
            elapsed = time.monotonic() - t0
            call_log.append({
                "task": task_type,
                "endpoint": ep_name,
                "model": model,
                "time": elapsed,
                "ok": True,
                "prompt_len": len(prompt),
                "response_len": len(result) if result else 0,
            })
            return result
        except Exception as e:
            elapsed = time.monotonic() - t0
            call_log.append({
                "task": task_type,
                "endpoint": ep_name,
                "model": model,
                "time": elapsed,
                "ok": False,
                "error": str(e)[:80],
            })
            raise

    router.generate = patched_generate

    messages = TEST_MESSAGES[:num_messages]
    total_start = time.monotonic()
    results = []

    for i, (role, text) in enumerate(messages):
        print(f"  --- Message {i + 1}/{len(messages)} ({role}) ---")
        print(f"  Text: {text[:70]}...")

        call_log.clear()
        msg_start = time.monotonic()
        try:
            memory_id = await processor.process_message(
                text=text,
                role=role,
                session_id=9999,  # diagnostic session
            )
            msg_elapsed = time.monotonic() - msg_start
            status = f"OK (memory_id={memory_id})" if memory_id else "SKIPPED"
        except Exception as e:
            msg_elapsed = time.monotonic() - msg_start
            status = f"FAILED: {e}"

        print(f"  Status: {status}")
        print(f"  Total time: {msg_elapsed:.2f}s")
        print(f"  LLM calls ({len(call_log)}):")
        for call in call_log:
            ok = "OK" if call["ok"] else f"FAIL: {call.get('error', '')}"
            print(f"    {call['task']:25s} -> {call['endpoint']:15s} "
                  f"({call['model']}) {call['time']:.2f}s [{ok}]")
            if call["ok"]:
                print(f"      prompt={call['prompt_len']} chars, response={call['response_len']} chars")

        # Calculate non-LLM time (SQLite + ChromaDB + regex tagging + noise check)
        llm_total = sum(c["time"] for c in call_log)
        non_llm = msg_elapsed - llm_total
        print(f"  Non-LLM time (sqlite+embed+tag+noise): {non_llm:.2f}s")
        print()

        results.append({
            "role": role,
            "time": msg_elapsed,
            "calls": list(call_log),
            "status": status,
            "llm_time": llm_total,
            "non_llm_time": non_llm,
        })

    router.generate = original_generate
    total_elapsed = time.monotonic() - total_start

    # Summary
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Total messages: {len(results)}")
    print(f"  Total time: {total_elapsed:.1f}s")
    if results:
        print(f"  Avg per message: {total_elapsed / len(results):.1f}s")
    print()

    # Breakdown by LLM step
    step_times: dict[str, list[float]] = {}
    step_endpoints: dict[str, set[str]] = {}
    step_failures: dict[str, int] = {}
    for r in results:
        for call in r["calls"]:
            task = call["task"]
            step_times.setdefault(task, []).append(call["time"])
            step_endpoints.setdefault(task, set()).add(call["endpoint"])
            if not call["ok"]:
                step_failures[task] = step_failures.get(task, 0) + 1

    print("  Per-step LLM breakdown:")
    for task, times in sorted(step_times.items()):
        avg = sum(times) / len(times)
        total = sum(times)
        endpoints = ", ".join(step_endpoints[task])
        fails = step_failures.get(task, 0)
        fail_str = f" ({fails} FAILURES)" if fails else ""
        print(f"    {task:25s} avg={avg:.1f}s  total={total:.1f}s  "
              f"calls={len(times)}  endpoints=[{endpoints}]{fail_str}")

    # Non-LLM time
    non_llm_times = [r["non_llm_time"] for r in results]
    if non_llm_times:
        print(f"\n  Non-LLM overhead (sqlite+embed+tag): avg={sum(non_llm_times)/len(non_llm_times):.2f}s  "
              f"total={sum(non_llm_times):.2f}s")

    # Bottleneck
    if step_times:
        bottleneck = max(step_times.items(), key=lambda x: sum(x[1]))
        print(f"\n  BOTTLENECK: {bottleneck[0]} "
              f"({sum(bottleneck[1]):.1f}s total, "
              f"{sum(bottleneck[1]) / len(bottleneck[1]):.1f}s avg)")

    # LLM calls per message
    calls_per_msg = [len(r["calls"]) for r in results]
    if calls_per_msg:
        print(f"\n  LLM calls per message: min={min(calls_per_msg)} max={max(calls_per_msg)} "
              f"avg={sum(calls_per_msg)/len(calls_per_msg):.1f}")
        print(f"  (2 = normal: summarize + rank.  3 = dedup triggered an extra REASONING call)")

    # Clean up diagnostic memories
    print("\n  Cleaning up diagnostic data...")
    try:
        memories = await sqlite.get_memories_by_session(9999)
        for mem in memories:
            try:
                chroma.delete_memory(mem.id)
            except Exception:
                pass
            await sqlite._db.execute("DELETE FROM memories WHERE id = ?", (mem.id,))
        await sqlite._db.commit()
        print(f"    Removed {len(memories)} diagnostic memories")
    except Exception as e:
        print(f"    Cleanup failed: {e}")

    await sqlite.close()


async def diagnose_concurrent_db(config):
    """Test concurrent SQLite access from two connections (reproduces 'database is locked').

    This simulates what happens in real use: the main thread and the worker
    thread both have their own SQLiteStore connections and both write.
    """
    print("=" * 70)
    print("CONCURRENT DB ACCESS TEST")
    print("=" * 70)
    print("  Simulates main thread + worker thread writing simultaneously")
    print()

    # Connection 1 (simulates main thread - session manager)
    sqlite1 = SQLiteStore(config.database.path)
    await sqlite1.initialize()

    # Connection 2 (simulates worker thread)
    sqlite2 = SQLiteStore(config.database.path)
    await sqlite2.initialize()

    # Create a test session
    session_id = await sqlite1.create_session(title="Diagnostic concurrent test", project=None)
    print(f"  Test session: {session_id}")

    errors = []
    timings = []

    async def writer_1(n: int):
        """Simulates main thread writes (session updates, message persistence)."""
        for i in range(n):
            t0 = time.monotonic()
            try:
                await sqlite1.update_session(
                    session_id,
                    last_active="2026-01-01T00:00:00",
                    message_count=i,
                )
                elapsed = time.monotonic() - t0
                timings.append(("writer1_session_update", elapsed))
            except Exception as e:
                elapsed = time.monotonic() - t0
                errors.append(("writer1", f"session update #{i}", str(e), elapsed))

            # Small delay to simulate real timing
            await asyncio.sleep(0.05)

    async def writer_2(n: int):
        """Simulates worker thread writes (create_memory + update_memory)."""
        from blipshell.models.memory import Memory, MemoryType
        from datetime import datetime, timezone

        for i in range(n):
            t0 = time.monotonic()
            try:
                memory = Memory(
                    session_id=session_id,
                    role="user",
                    content=f"Concurrent test message {i}",
                    summary=f"Test summary {i}",
                    timestamp=datetime.now(timezone.utc),
                    memory_type=MemoryType.CONVERSATION,
                )
                mem_id = await sqlite2.create_memory(memory)
                elapsed_create = time.monotonic() - t0
                timings.append(("writer2_create_memory", elapsed_create))

                # Simulate the rank+importance update that comes after LLM call
                t1 = time.monotonic()
                await sqlite2.update_memory(mem_id, rank=3, importance=0.5)
                elapsed_update = time.monotonic() - t1
                timings.append(("writer2_update_memory", elapsed_update))
            except Exception as e:
                elapsed = time.monotonic() - t0
                errors.append(("writer2", f"memory #{i}", str(e), elapsed))

            await asyncio.sleep(0.02)

    # Test 1: Sequential (baseline)
    print("  Test 1: Sequential writes (baseline)...")
    errors.clear()
    timings.clear()
    t0 = time.monotonic()
    await writer_1(10)
    await writer_2(10)
    elapsed = time.monotonic() - t0
    print(f"    Time: {elapsed:.2f}s  Errors: {len(errors)}")
    if timings:
        avg = sum(t[1] for t in timings) / len(timings)
        print(f"    Avg write: {avg*1000:.1f}ms")

    # Test 2: Concurrent (reproduces the real scenario)
    print("\n  Test 2: Concurrent writes (simulates main + worker)...")
    errors.clear()
    timings.clear()
    t0 = time.monotonic()
    await asyncio.gather(writer_1(20), writer_2(20))
    elapsed = time.monotonic() - t0
    print(f"    Time: {elapsed:.2f}s  Errors: {len(errors)}")
    if errors:
        print(f"    ERRORS FOUND:")
        for source, op, err, dur in errors:
            print(f"      [{source}] {op}: {err} (after {dur:.2f}s)")
    else:
        print(f"    No errors - concurrent writes worked fine")
    if timings:
        avg = sum(t[1] for t in timings) / len(timings)
        max_t = max(t[1] for t in timings)
        print(f"    Avg write: {avg*1000:.1f}ms  Max write: {max_t*1000:.1f}ms")

    # Test 3: Heavy concurrent (stress test)
    print("\n  Test 3: Heavy concurrent writes (stress test, 50 each)...")
    errors.clear()
    timings.clear()
    t0 = time.monotonic()
    await asyncio.gather(writer_1(50), writer_2(50))
    elapsed = time.monotonic() - t0
    print(f"    Time: {elapsed:.2f}s  Errors: {len(errors)}")
    if errors:
        print(f"    {len(errors)} ERRORS:")
        for source, op, err, dur in errors[:5]:
            print(f"      [{source}] {op}: {err} (after {dur:.2f}s)")
        if len(errors) > 5:
            print(f"      ... and {len(errors) - 5} more")
    else:
        print(f"    No errors - concurrent writes worked fine")
    if timings:
        avg = sum(t[1] for t in timings) / len(timings)
        max_t = max(t[1] for t in timings)
        slow = [t for t in timings if t[1] > 1.0]
        print(f"    Avg write: {avg*1000:.1f}ms  Max write: {max_t*1000:.1f}ms")
        if slow:
            print(f"    Slow writes (>1s): {len(slow)}")

    # Cleanup
    print("\n  Cleaning up...")
    try:
        memories = await sqlite1.get_memories_by_session(session_id)
        for mem in memories:
            await sqlite1._db.execute("DELETE FROM memories WHERE id = ?", (mem.id,))
        await sqlite1._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await sqlite1._db.commit()
        print(f"    Cleaned up session {session_id} and {len(memories)} memories")
    except Exception as e:
        print(f"    Cleanup failed: {e}")

    await sqlite1.close()
    await sqlite2.close()


async def diagnose_concurrent_with_llm(config):
    """Test what actually happens: LLM calls + DB writes concurrently.

    The real scenario is: worker is mid-process_message (LLM call taking 5-15s)
    while main thread is also writing (session updates, message persistence).
    The LLM call itself doesn't hold a DB lock, but the write AFTER the LLM
    call might collide with a main-thread write.
    """
    print("=" * 70)
    print("CONCURRENT LLM + DB TEST (simulates real workload)")
    print("=" * 70)

    sqlite1 = SQLiteStore(config.database.path)
    await sqlite1.initialize()
    sqlite2 = SQLiteStore(config.database.path)
    await sqlite2.initialize()

    endpoint_mgr = EndpointManager(config.endpoints, config.llm)
    router = LLMRouter(config.models, endpoint_mgr)

    try:
        from blipshell.memory.chroma_store import ChromaStore
        chroma = ChromaStore()
    except Exception as e:
        print(f"  ChromaDB unavailable: {e}")
        await sqlite1.close()
        await sqlite2.close()
        return

    from blipshell.memory.processor import MemoryProcessor
    processor = MemoryProcessor(
        sqlite2, chroma, router,
        config=config.memory,
        max_tags=config.tagging.max_tags,
    )

    session_id = await sqlite1.create_session(title="Diagnostic concurrent LLM test")
    print(f"  Test session: {session_id}")
    print(f"  Running 3 messages through pipeline WHILE main thread writes...")
    print()

    errors = []

    async def main_thread_writes():
        """Simulates what the main thread does during a conversation."""
        for i in range(30):
            try:
                await sqlite1.update_session(
                    session_id,
                    last_active="2026-01-01T00:00:00",
                    message_count=i,
                )
                # Also simulate message persistence
                await sqlite1.save_session_message(
                    session_id, "user", f"Test message {i}", "2026-01-01T00:00:00",
                )
            except Exception as e:
                errors.append(("main_thread", f"write #{i}", str(e)))
            await asyncio.sleep(0.3)

    async def worker_pipeline():
        """Simulates worker processing messages through full pipeline."""
        for i, (role, text) in enumerate(TEST_MESSAGES[:3]):
            t0 = time.monotonic()
            try:
                memory_id = await processor.process_message(
                    text=text, role=role, session_id=session_id,
                )
                elapsed = time.monotonic() - t0
                status = f"OK (id={memory_id})" if memory_id else "SKIPPED"
                print(f"  Worker msg {i+1}: {status} in {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.monotonic() - t0
                errors.append(("worker", f"message #{i}", str(e)))
                print(f"  Worker msg {i+1}: FAILED in {elapsed:.1f}s: {e}")

    t0 = time.monotonic()
    await asyncio.gather(main_thread_writes(), worker_pipeline())
    total = time.monotonic() - t0

    print(f"\n  Total time: {total:.1f}s")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for source, op, err in errors:
            print(f"    [{source}] {op}: {err}")
    else:
        print(f"  No errors — concurrent LLM + DB worked fine")

    # Cleanup
    print("\n  Cleaning up...")
    try:
        memories = await sqlite1.get_memories_by_session(session_id)
        for mem in memories:
            try:
                chroma.delete_memory(mem.id)
            except Exception:
                pass
            await sqlite1._db.execute("DELETE FROM memories WHERE id = ?", (mem.id,))
        await sqlite1._db.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
        await sqlite1._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await sqlite1._db.commit()
        print(f"    Cleaned up")
    except Exception as e:
        print(f"    Cleanup failed: {e}")

    await sqlite1.close()
    await sqlite2.close()


async def main():
    parser = argparse.ArgumentParser(description="Diagnose memory pipeline performance")
    parser.add_argument("--messages", "-m", type=int, default=5,
                        help="Number of test messages to process (default: 5)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--routing-only", action="store_true",
                        help="Only show routing map, don't run pipeline")
    parser.add_argument("--concurrent-only", action="store_true",
                        help="Only run concurrent DB tests")
    parser.add_argument("--skip-concurrent", action="store_true",
                        help="Skip concurrent DB tests")
    args = parser.parse_args()

    config_mgr = ConfigManager(args.config)
    config = config_mgr.config

    print("=" * 70)
    print("  BLIPSHELL MEMORY PIPELINE DIAGNOSTIC")
    print("=" * 70)
    print(f"  Config: {config_mgr.config_path}")
    print(f"  Database: {config.database.path}")
    print()

    if args.concurrent_only:
        await diagnose_concurrent_db(config)
        await diagnose_concurrent_with_llm(config)
        return

    endpoint_mgr = EndpointManager(config.endpoints, config.llm)
    router = LLMRouter(config.models, endpoint_mgr)

    await diagnose_routing(router, endpoint_mgr)

    if args.routing_only:
        return

    await diagnose_individual_calls(router, endpoint_mgr)
    await diagnose_full_pipeline(config, args.messages)

    if not args.skip_concurrent:
        await diagnose_concurrent_db(config)
        await diagnose_concurrent_with_llm(config)


if __name__ == "__main__":
    asyncio.run(main())
