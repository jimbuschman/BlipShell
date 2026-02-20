"""Quick debug script to check why process_message returns None."""
import asyncio
from unittest.mock import AsyncMock, MagicMock
from tests.conftest import _canned_generate
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import MemoryConfig
import tempfile, os

async def main():
    # Test the mock router
    router = MagicMock()
    router.generate = AsyncMock(side_effect=_canned_generate)

    r1 = await router.generate("summarization", "test prompt", system="test system")
    print(f"summarization mock returns: {repr(r1)}")

    r2 = await router.generate("ranking_importance", "test prompt", system="test system")
    print(f"ranking_importance mock returns: {repr(r2)}")

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
            sqlite=store, chroma=chroma, router=router, config=config,
        )

        session_id = await store.create_session("Test")
        print(f"session_id: {session_id}")

        text = "I think Python performance tuning with cProfile and line_profiler tools is what I want to explore next for my project"
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

asyncio.run(main())
