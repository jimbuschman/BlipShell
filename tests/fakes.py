"""Reusable test fakes for driving the REAL ChatLoop without Ollama.

The point: validate end-to-end loop *wiring* (completion detection, guardrails
gating, look-before-review, dedup, tool execution) deterministically on the dev
box, instead of deferring it all to manual Ollama-PC checks. The model is
scripted; everything else (ChatLoop, ToolRegistry, GuardrailsEngine) is real.
"""

from blipshell.core.tools.base import Tool, ToolRegistry
from blipshell.models.tools import ToolDefinition


class ScriptedLLMClient:
    """A fake LLM client that feeds canned responses into the real ChatLoop.

    `script` is a list of turns, consumed one per chat_stream/chat call:
      - {"tools": [(name, args_dict), ...], "text": "optional"}  -> tool calls
      - {"text": "..."}                                          -> text-only (loop ends)
    Past the end of the script it returns empty text (loop terminates).

    Records a shallow snapshot of the messages list it was handed on each call
    in `self.sent_messages` (so tests can assert on the system prompt etc.).
    """

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.calls = 0
        self.sent_messages: list[list[dict]] = []

    def _next_chunk(self, messages):
        self.calls += 1
        self.sent_messages.append([dict(m) for m in messages])
        if self._i >= len(self._script):
            return {"message": {"content": "", "tool_calls": None}, "done": True}
        turn = self._script[self._i]
        self._i += 1
        tool_calls = None
        if turn.get("tools"):
            tool_calls = [
                {"function": {"name": n, "arguments": a}, "id": f"tc{idx}"}
                for idx, (n, a) in enumerate(turn["tools"])
            ]
        return {
            "message": {"content": turn.get("text", ""), "tool_calls": tool_calls},
            "done": True,
        }

    async def chat_stream(self, messages, model, tools=None, **kwargs):
        yield self._next_chunk(messages)

    async def chat(self, messages, model, tools=None, **kwargs):
        return self._next_chunk(messages)


class FakeTool(Tool):
    """Minimal executable tool — returns a fixed string."""

    def __init__(self, name: str, read_only: bool = False, result: str | None = None):
        self._name = name
        self.read_only = read_only
        self._result = result if result is not None else f"{name} executed"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self._name, description=f"fake {self._name} tool")

    async def execute(self, **kwargs) -> str:
        return self._result


class RecordingRouter:
    """Stub router counting generate() calls (to assert LLM-call avoidance)."""

    def __init__(self, result: str = "PASS"):
        self.calls = 0
        self._result = result

    async def generate(self, *args, **kwargs):
        self.calls += 1
        return self._result


def make_registry(*tools: Tool) -> ToolRegistry:
    """A registry with the given fake tools (no approval callback → tools run)."""
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


class ThoughtHarness:
    """Real SQLiteStore behind the self-thought API, with seed/read helpers.

    Self-thoughts moved from one app_metadata JSON row to a real table
    (2026-08-07). The tests used to seed and assert against that JSON blob
    directly, so they were coupled to the storage format — and a fake
    reimplementing the new SQL semantics would leave the actual SQL, which is
    where the new bugs are, untested. This wraps the REAL store instead.
    """

    def __init__(self, sqlite):
        self.sqlite = sqlite

    async def seed(self, items: list[dict]) -> list[int]:
        """Set the store to EXACTLY these thoughts, in order.

        Replaces rather than appends, matching the semantics of the JSON blob
        these tests were written against (`meta.data[KEY] = json.dumps(...)`)
        — several do a read-modify-write of the whole list. Deleting is fine
        here; the archive-never-delete mandate is about the live store, and a
        seeding helper that archived would leave rows visible to
        include_archived assertions.
        """
        await self.sqlite._db.execute("DELETE FROM self_thoughts")
        await self.sqlite._db.commit()
        ids = []
        for it in items:
            ids.append(await self.sqlite.add_self_thought(
                it["text"],
                it.get("created_at") or "2026-01-01T00:00:00+00:00",
                embedding=it.get("embedding"),
                weight=it.get("weight", 1.0),
                surfaced=bool(it.get("surfaced")),
                echo_count=it.get("echo_count", 0),
                surface_count=it.get("surface_count", 0),
            ))
        return ids

    async def rows(self, *, include_archived: bool = False) -> list[dict]:
        return await self.sqlite.get_self_thoughts(
            with_embeddings=True, include_archived=include_archived,
        )

    async def texts(self) -> list[str]:
        return [r["text"] for r in await self.rows()]
