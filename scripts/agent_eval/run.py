"""Deep agentic evaluation of Ollama free-tier cloud models for the MAIN
(tool-calling) role in BlipShell.

Real BlipShell tool schemas, real BlipShell system prompt, a multi-turn loop
shaped like ChatLoop, against a simulated repo. Produces full transcripts for
human grading plus objective per-scenario signals.

Usage: python run.py <model> [episodes_per_scenario]
"""

import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(r"C:\Users\[user]\source\repos\jimbuschman\BlipShell")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import workspace as w  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402

from blipshell.models.config import AgentConfig  # noqa: E402

import os  # noqa: E402

# Override for the Ollama PC over Tailscale when testing local small models.
OLLAMA_URL = os.environ.get("DEEPTEST_URL", "http://localhost:11434")
# The ollama SDK defaults to timeout=None (httpx waits forever). A small model
# that wedges would hang the whole sweep, so cap every call.
CALL_TIMEOUT = float(os.environ.get("DEEPTEST_TIMEOUT", "300"))
# BlipShell's real ceiling is AgentConfig.max_tool_iterations = 50. At 8 the
# deepest scenario (diagnose_and_fix) was being truncated mid-work, which
# measures the harness, not the model.
MAX_TURNS = 20
TOOLS = json.load(open(HERE / "real_tools.json", encoding="utf-8"))
SYSTEM = AgentConfig().system_prompt


def _extract(resp):
    """(content, tool_calls) from an ollama chat response.

    The SDK returns a pydantic ChatResponse, not a dict — dict-only access
    silently yields empty content and zero tool calls for every turn.
    """
    msg = resp.get("message") if isinstance(resp, dict) else getattr(resp, "message", None)
    if msg is None:
        return "", []
    if isinstance(msg, dict):
        return msg.get("content") or "", msg.get("tool_calls") or []
    return getattr(msg, "content", "") or "", list(getattr(msg, "tool_calls", None) or [])


def _call_info(tc):
    fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
    if fn is None:
        return "", {}
    if isinstance(fn, dict):
        name, args = fn.get("name") or "", fn.get("arguments")
    else:
        name, args = getattr(fn, "name", "") or "", getattr(fn, "arguments", None)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"_raw": args}
    if args is not None and not isinstance(args, dict):
        try:
            args = dict(args)
        except Exception:
            args = {"_raw": str(args)}
    return name, (args or {})


def _to_wire(tc):
    """Plain-dict form of a tool call, for sending back in the message history."""
    name, args = _call_info(tc)
    return {"function": {"name": name, "arguments": args}}


async def run_episode(client, model, scenario):
    w.reset()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": scenario["user"]},
    ]
    transcript = {"scenario": scenario["id"], "model": model, "turns": 0,
                  "calls": [], "results": [], "final_text": "", "error": None,
                  "hit_turn_limit": False, "empty_final": False, "wire": []}
    start = time.perf_counter()
    try:
        for turn in range(MAX_TURNS):
            transcript["turns"] = turn + 1
            resp = await asyncio.wait_for(
                client.chat(model=model, messages=messages, tools=TOOLS),
                timeout=CALL_TIMEOUT)
            content, tcs = _extract(resp)
            transcript["wire"].append({"turn": turn + 1, "text": content,
                                       "tool_calls": [_call_info(t) for t in tcs]})
            if not tcs:
                transcript["final_text"] = content
                break
            messages.append({"role": "assistant", "content": content,
                             "tool_calls": [_to_wire(t) for t in tcs]})
            for tc in tcs:
                name, args = _call_info(tc)
                transcript["calls"].append({"name": name, "args": args})
                if name in w.DISPATCH:
                    try:
                        result = w.DISPATCH[name](args)
                    except Exception as e:
                        result = "Error: %s" % e
                elif name == "task_complete":
                    result = "OK: task marked complete"
                elif name == "ask_user":
                    # Answer once so the loop can continue, as a real user would.
                    result = "User: the failing test in tests/test_utils.py."
                else:
                    result = "Error: unknown tool %s" % name
                transcript["results"].append(result)
                messages.append({"role": "tool", "name": name, "content": result})
            if all(_call_info(t)[0] == "task_complete" for t in tcs):
                transcript["final_text"] = content
                break
        else:
            transcript["hit_turn_limit"] = True
            transcript["final_text"] = content
        # A model that stops with no prose has answered nothing, even if its
        # tool calls were fine. Fall back to the last text it did produce so
        # the content checks grade what it actually said, and flag the gap.
        if not (transcript["final_text"] or "").strip():
            transcript["empty_final"] = True
            for wt in reversed(transcript["wire"]):
                if (wt.get("text") or "").strip():
                    transcript["final_text"] = wt["text"]
                    break
    except Exception as e:
        transcript["error"] = "%s: %s" % (type(e).__name__, e)
    transcript["elapsed_s"] = round(time.perf_counter() - start, 2)
    try:
        transcript["signals"] = scenario["check"](transcript)
    except Exception as e:
        transcript["signals"] = {"check_error": str(e)}
    return transcript


async def main():
    model = sys.argv[1]
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    from ollama import AsyncClient
    client = AsyncClient(host=OLLAMA_URL)

    out = []
    for scenario in SCENARIOS:
        for ep in range(episodes):
            t = await run_episode(client, model, scenario)
            t["episode"] = ep
            out.append(t)
            sig = t.get("signals", {})
            flag = "ERR" if t["error"] else ("LIM" if t["hit_turn_limit"] else "ok ")
            print("  %s %-22s ep%d turns=%d %s" % (
                flag, scenario["id"], ep, t["turns"],
                " ".join("%s=%s" % (k, v) for k, v in sig.items())))

    safe = model.replace(":", "_").replace("/", "_")
    path = HERE / ("transcripts__%s.json" % safe)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print("saved %s (%d episodes)" % (path.name, len(out)))


asyncio.run(main())
