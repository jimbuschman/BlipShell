"""Record every candidate call a benchmark run makes.

The old harness scored outputs and DISCARDED them — the result file's `raw`
field was null on almost every row. That was fine when nobody read
transcripts; the whole point of the 2026-08-10 rethink is that transcripts
are now the primary artifact (scores are the index, reading is the
verdict), so every prompt/response pair the candidate produces is captured
here, at the one chokepoint every suite already goes through: the pinned
candidate router.

Also the speed story: wall-clock per call plus an estimated tok/s
(chars/4/elapsed — an approximation, labeled as such; the router API
doesn't surface Ollama's eval_count). Cold-load shows up naturally as the
first call's outlier latency.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Truncation caps: transcripts are for READING and land in a committed file.
# Prompts repeat per-suite boilerplate, so they get the tighter cap.
MAX_PROMPT_CHARS = 4000
MAX_RESPONSE_CHARS = 8000


def _trunc(text: Optional[str], cap: int) -> Optional[str]:
    if text is None:
        return None
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated — {len(text)} chars total]"


class RecordingRouter:
    """Wraps the candidate router; records every generate() call.

    Everything else delegates to the wrapped router untouched, so the
    harness and the reused suite runners can't tell the difference.
    """

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[dict] = []
        # Set by the harness as it moves between suites; "?" means a call
        # happened outside any labeled section (worth noticing in analysis).
        self.suite = "?"
        self.repeat = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def generate(self, task_type, prompt="", system=None, **kwargs):
        t0 = time.monotonic()
        response, error = None, None
        try:
            response = await self._inner.generate(
                task_type, prompt, system=system, **kwargs,
            )
            return response
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            elapsed = time.monotonic() - t0
            est_tokens = (len(response) // 4) if response else 0
            self.calls.append({
                "suite": self.suite,
                "repeat": self.repeat,
                "task_type": str(getattr(task_type, "value", task_type)),
                "system": _trunc(system, MAX_PROMPT_CHARS),
                "prompt": _trunc(prompt, MAX_PROMPT_CHARS),
                "response": _trunc(response, MAX_RESPONSE_CHARS),
                "error": error,
                "elapsed_s": round(elapsed, 2),
                # chars/4 per second — an APPROXIMATION of tok/s (the router
                # API doesn't surface Ollama's eval_count). Comparable across
                # calls in the same run; don't quote it as a hardware number.
                "est_tok_s": round(est_tokens / elapsed, 1) if elapsed > 0 else None,
            })


class RecordingClient:
    """Wraps an LLM client's chat() — the path router.generate() never sees.

    The pilot proved the gap: the agentic coding suite and tool-calling chat
    go through client.chat with tools, so the suite whose failures most need
    reading (coding_agentic 0.19) produced zero transcripts. This records at
    the client layer; `recorder` is the run's RecordingRouter, so suite and
    repeat labels stay consistent across both layers.
    """

    def __init__(self, inner, recorder):
        self._inner = inner
        self._recorder = recorder

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def chat(self, *args, **kwargs):
        t0 = time.monotonic()
        resp, error = None, None
        try:
            resp = await self._inner.chat(*args, **kwargs)
            return resp
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            msgs = kwargs.get("messages")
            if msgs is None:
                msgs = next((a for a in args if isinstance(a, list)), [])
            last_user = next(
                (str(m.get("content"))[:MAX_PROMPT_CHARS]
                 for m in reversed(msgs) if m.get("role") == "user"), "",
            )
            message = (resp or {}).get("message") or {}
            tool_calls = [
                (tc.get("function") or {}).get("name")
                for tc in (message.get("tool_calls") or [])
            ][:10]
            elapsed = time.monotonic() - t0
            self._recorder.calls.append({
                "suite": self._recorder.suite,
                "repeat": self._recorder.repeat,
                "task_type": "chat",
                "system": None,
                "prompt": last_user,
                "response": _trunc(str(message.get("content") or ""), MAX_RESPONSE_CHARS),
                "tool_calls": tool_calls or None,
                "error": error,
                "elapsed_s": round(elapsed, 2),
                "est_tok_s": None,
            })


def wrap_router_clients(router) -> None:
    """Wrap every endpoint client behind a RecordingRouter with RecordingClient.

    Call AFTER building the RecordingRouter; idempotent enough for one run.
    """
    manager = getattr(router, "_endpoint_manager", None)
    for ep in getattr(manager, "_endpoints", []) or []:
        if ep.client is not None and not isinstance(ep.client, RecordingClient):
            ep.client = RecordingClient(ep.client, router)
