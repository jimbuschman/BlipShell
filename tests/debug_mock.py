"""Quick debug script to check why process_message returns None in pytest."""
import asyncio
import sys
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.noise import should_skip_memory
from blipshell.models.config import MemoryConfig


def _canned_generate(task_type, prompt="", system=None, think=None):
    print(f"  [mock] generate called: task_type={task_type!r}")
    if task_type == "summarization":
        return "User discussed Python performance tuning and optimization strategies."
    if task_type == "ranking_importance":
        return "3 0.5"
    if task_type == "reasoning":
        return "When discussing performance, focus on profiling before optimizing."
    return "test response"


async def main():
    # Check versions
    try:
        import pytest_asyncio
        print(f"pytest-asyncio version: {pytest_asyncio.__version__}")
    except Exception as e:
        print(f"pytest-asyncio: {e}")
    try:
        import pytest
        print(f"pytest version: {pytest.__version__}")
    except Exception as e:
        print(f"pytest: {e}")
    print(f"Python: {sys.version}")
    print()

    # Test noise filter with the EXACT test input
    text = "I think Python performance tuning with cProfile and line_profiler tools is what I want to explore next for my project"
    print(f"Noise check: len={len(text)}, skip={should_skip_memory(text)}")
    print()

    # Test the mock router
    router = MagicMock()
    router.generate = AsyncMock(side_effect=_canned_generate)

    r1 = await router.generate("summarization", "test prompt", system="test system")
    print(f"summarization mock returns: {repr(r1)}")

    r2 = await router.generate("ranking_importance", "test prompt", system="test system")
    print(f"ranking_importance mock returns: {repr(r2)}")
    print()

    # Test full pipeline
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        store = SQLiteStore(db_path)
        await store.initialize()

        chroma = MagicMock()
        chroma.add_memory = MagicMock()
        chroma.search_core_memories.return_value = []

        config = MemoryConfig(
            total_context_tokens=4096,
            system_prompt_reserve=256,
        )
        processor = MemoryProcessor(
            sqlite=store, vectors=chroma, router=router, config=config,
        )

        session_id = await store.create_session("Test")
        print(f"session_id: {session_id}")

        print("Calling process_message...")
        mem_id = await processor.process_message(
            text=text, role="user", session_id=session_id,
        )
        print(f"mem_id: {mem_id}")

        if mem_id is None:
            print("FAILED: mem_id is None")
        else:
            mem = await store.get_memory(mem_id)
            print(f"summary: {mem.summary}")
            print(f"rank: {mem.rank}, importance: {mem.importance}")

        await store.close()
    finally:
        os.unlink(db_path)

    # Now test the conftest version
    print("\n--- Testing conftest _canned_generate ---")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from conftest import _canned_generate as conftest_gen
    router2 = MagicMock()
    router2.generate = AsyncMock(side_effect=conftest_gen)
    r = await router2.generate("summarization", "test", system="sys")
    print(f"conftest summarization: {repr(r)}")
    r = await router2.generate("ranking_importance", "test", system="sys")
    print(f"conftest ranking_importance: {repr(r)}")

asyncio.run(main())
