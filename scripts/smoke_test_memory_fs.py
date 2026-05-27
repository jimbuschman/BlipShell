"""End-to-end smoke test for the memory filesystem feature.

Runs without an LLM — exercises registration + tool dispatch + approval +
vector-sync + notes persistence using a real temp SQLiteStore and a mock
VectorStore.

Validates:
    1. Backends instantiate and all 4 tools register
    2. memory_view returns the tier listing
    3. Lessons are READ-ONLY (read works; create/edit/delete blocked)
    4. Core create triggers approval + embeds (vector sync)
    5. Denying approval blocks the core write (no embed)
    6. Core edit re-embeds; core delete deactivates + un-embeds
    7. Notes round-trip (create/read), shared with session_notes store
    8. Read-only tiers (digests/sessions/friction) reject writes

Exits 0 on success, 1 on any failure.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from blipshell.core.tools.base import ToolRegistry
from blipshell.core.tools.memory_fs import (
    MemoryCreateTool,
    MemoryDeleteTool,
    MemoryStrReplaceTool,
    MemoryViewTool,
)
from blipshell.memory.fs_backend import MemoryFSBackend
from blipshell.memory.fs_notes import NotesBackend
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.memory import Lesson


CHECKS_RUN = 0
CHECKS_PASSED = 0
CHECKS_FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS_RUN, CHECKS_PASSED
    CHECKS_RUN += 1
    if condition:
        CHECKS_PASSED += 1
        print(f"  [PASS] {name}")
    else:
        msg = name + (f" — {detail}" if detail else "")
        CHECKS_FAILED.append(msg)
        print(f"  [FAIL] {msg}")


async def run():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        store = SQLiteStore(db_path)
        await store.initialize()
        session_id = await store.create_session(title="Smoke session")

        vectors = MagicMock()
        vectors.add_core_memory = MagicMock()
        vectors.delete_core_memory = MagicMock()

        backend = MemoryFSBackend(store, vectors)
        shared_notes: dict[str, str] = {}
        notes = NotesBackend(store, shared_notes, lambda: session_id, max_notes=10)

        approval = {"value": True}

        async def approval_cb(name, args):
            return approval["value"]

        registry = ToolRegistry()
        registry.register(MemoryViewTool(backend, notes), group="memory_fs")
        registry.register(MemoryCreateTool(backend, notes, approval_cb), group="memory_fs")
        registry.register(MemoryStrReplaceTool(backend, notes, approval_cb), group="memory_fs")
        registry.register(MemoryDeleteTool(backend, notes, approval_cb), group="memory_fs")

        view = registry._tools["memory_view"]
        create = registry._tools["memory_create"]
        edit = registry._tools["memory_str_replace"]
        delete = registry._tools["memory_delete"]

        # 1. tier listing
        print("\n[1] Tier listing")
        out = await view.execute(path="/memories")
        check("Lists lessons", "/memories/lessons/" in out)
        check("Lists core", "/memories/core/" in out)
        check("Lists notes", "/memories/notes/" in out)

        # 2. lessons read-only
        print("\n[2] Lessons are read-only")
        lid = await store.create_lesson(Lesson(content="Never dig straight down", project="mc"))
        body = await view.execute(path=f"/memories/lessons/mc/{lid}-x.md")
        check("Lesson reads", body == "Never dig straight down", detail=body)
        blocked = await create.execute(path="/memories/lessons/mc/", content="new")
        check("Lesson create blocked", "read-only" in blocked.lower(), detail=blocked)
        blocked = await edit.execute(
            path=f"/memories/lessons/mc/{lid}-x.md", old_text="dig", new_text="mine")
        check("Lesson edit blocked", "read-only" in blocked.lower(), detail=blocked)
        blocked = await delete.execute(path=f"/memories/lessons/mc/{lid}-x.md")
        check("Lesson delete blocked", "read-only" in blocked.lower(), detail=blocked)

        # 3. core create with approval + embed
        print("\n[3] Core create — approval + vector sync")
        approval["value"] = False
        denied = await create.execute(path="/memories/core/", content="User likes dark mode.")
        check("Core denied when approval=NO", "denied" in denied.lower(), detail=denied)
        check("No embed on denial", vectors.add_core_memory.call_count == 0)

        approval["value"] = True
        granted = await create.execute(path="/memories/core/", content="User likes dark mode.")
        check("Core create succeeds", granted.startswith("Created /memories/core/"), detail=granted)
        check("Embedded on create", vectors.add_core_memory.call_count == 1)
        canonical = granted.removeprefix("Created ").strip()

        # 4. core edit re-embeds
        print("\n[4] Core edit re-embeds")
        vectors.add_core_memory.reset_mock()
        await edit.execute(path=canonical, old_text="dark", new_text="light")
        updated = await view.execute(path=canonical)
        check("Edit applied", "light mode" in updated, detail=updated)
        check("Re-embedded on edit", vectors.add_core_memory.call_count == 1)

        # 5. core delete deactivates + un-embeds
        print("\n[5] Core delete — deactivate + un-embed")
        del_out = await delete.execute(path=canonical)
        check("Core delete succeeds", del_out.startswith("Deleted "), detail=del_out)
        check("Un-embedded on delete", vectors.delete_core_memory.call_count == 1)

        # 6. notes round-trip + shared with session_notes
        print("\n[6] Notes round-trip + shared store")
        ncreate = await create.execute(path="/memories/notes/plan.md", content="Refactor auth.")
        check("Note create", "Created /memories/notes/plan.md" in ncreate, detail=ncreate)
        nbody = await view.execute(path="/memories/notes/plan.md")
        check("Note reads back", nbody == "Refactor auth.", detail=nbody)
        check("Note in shared dict", shared_notes.get("plan") == "Refactor auth.")
        persisted = await store.get_session_notes(session_id)
        check("Note persisted to session_notes", persisted.get("plan") == "Refactor auth.")

        # 7. read-only tiers reject writes
        print("\n[7] Read-only tiers reject writes")
        d = await create.execute(path="/memories/digests/blipshell.md", content="manual")
        check("Digest create blocked", "read-only" in d.lower(), detail=d)
        s = await delete.execute(path="/memories/sessions/1-x.md")
        check("Session delete blocked", "read-only" in s.lower(), detail=s)

        await store.close()
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def main():
    print("Memory Filesystem Smoke Test (revised design)")
    print("=" * 60)
    try:
        asyncio.run(run())
    except Exception:
        print("\n[FATAL] Unhandled exception:")
        traceback.print_exc()
        return 2

    print("\n" + "=" * 60)
    print(f"{CHECKS_PASSED}/{CHECKS_RUN} checks passed")
    if CHECKS_FAILED:
        print("\nFailed checks:")
        for f in CHECKS_FAILED:
            print(f"  - {f}")
        return 1
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
