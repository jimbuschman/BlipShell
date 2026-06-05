"""Chat pipeline mixin for Agent.

Extracts the main chat entry point, simple/planned paths, memory search,
message building, and event logging.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    pass  # All types accessed via self

from blipshell.core.executor import build_executor_narrative
from blipshell.llm.exceptions import is_model_error
from blipshell.llm.router import TaskType
from blipshell.memory.manager import PoolItem, estimate_tokens
from blipshell.memory.query_profiles import classify_query, compute_pool_budgets
from blipshell.models.session import MessageRole

logger = logging.getLogger(__name__)


class ChatMixin:
    """Chat pipeline methods mixed into Agent."""

    async def chat(
        self,
        user_message: str,
        on_token: Optional[Callable[[str], None]] = None,
        force_plan: bool = False,
        on_tool_display: Optional[Callable] = None,
        research_mode: bool = False,
    ) -> str:
        """Process a user message through the full agent pipeline.

        Routes between simple chat and planned execution based on
        complexity classification.

        Args:
            user_message: The user's input
            on_token: Optional callback for streaming tokens
            force_plan: If True, skip classification and go straight to planning
            on_tool_display: Optional callback for structured tool batch display
            research_mode: If True, boost tool budget and inject research guidance

        Returns:
            The assistant's complete response
        """
        if not hasattr(self, 'session_manager') or self.session_manager is None:
            raise RuntimeError("No active session — call start_session() before chat()")

        # Sanitize surrogate characters — Windows console/clipboard can produce
        # unpaired UTF-16 surrogates (U+D800–U+DFFF) that crash UTF-8 encoding
        # downstream (LLM clients, ChromaDB, SQLite, prompt_toolkit history).
        user_message = "".join(
            c if ord(c) < 0xD800 or ord(c) > 0xDFFF else "\ufffd"
            for c in user_message
        )

        # Track user activity (and the gap since last turn, for "you're back").
        import time
        _now = time.time()
        _activity_gap = _now - self._last_user_activity
        self._last_user_activity = _now

        # Reset tool call tracking (populated by _chat_simple/_chat_planned)
        self._last_tool_calls = []

        # Guardrails: correction detection (cheap regex, no LLM call)
        if (hasattr(self, 'config') and self.config.guardrails.enabled
                and self.config.guardrails.correction_detector):
            await self._detect_and_persist_correction(user_message)

        # Add user message to session
        self.session_manager.add_message(MessageRole.USER, user_message)

        # Decide execution path — only force_plan triggers the executor.
        # The model handles complexity naturally through the unified chat loop.
        needs_planning = force_plan

        # Event: turn_start
        self._turn_number += 1
        session_id = self.session_manager.session_id
        await self._log_event("turn_start", {
            "query_length": len(user_message),
            "route": "planned" if needs_planning else "simple",
        })

        # Update the affective interior from this turn — it drives the cube
        # "face" (display-only; never changes the response).
        if _activity_gap > 600:   # back after ~10+ minutes away
            await self._update_mood("user_returned")
        await self._update_mood("interaction")

        if needs_planning:
            logger.info("Message classified as complex — using planned execution")
            response = await self._chat_planned(user_message, on_token=on_token, on_tool_display=on_tool_display)
        else:
            logger.info("Message classified as simple — using direct chat")
            response = await self._chat_simple(user_message, on_token=on_token, on_tool_display=on_tool_display, research_mode=research_mode)

        # Self-reflection: second LLM pass to catch errors/gaps
        if self.reflect_enabled and response and not response.startswith("Error:"):
            if on_token:
                on_token("\n\n\x1b[2m[Reflecting...]\x1b[0m\n\n")
            response = await self._reflect_on_response(user_message, response, on_token)

        # Add assistant response to session
        if response and response.strip():
            self.session_manager.add_message(MessageRole.ASSISTANT, response)
        else:
            # Empty response — add placeholder so session continuity isn't broken
            logger.warning("LLM returned empty response for: %s", user_message[:80])
            response = "[No response generated]"
            self.session_manager.add_message(MessageRole.ASSISTANT, response)

        # Background: dump to memory periodically (tracked for clean shutdown)
        task = asyncio.create_task(self._background_memory_processing())
        self._background_tasks.add(task)

        def _on_task_done(t):
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                logger.error("Background memory processing failed: %s", exc)

        task.add_done_callback(_on_task_done)

        return response

    async def _detect_and_persist_correction(self, user_message: str):
        """Detect if a user message is correcting the assistant and persist as anti-pattern lesson.

        Runs on every user message when guardrails.correction_detector is enabled.
        Uses cheap regex matching — no LLM call.
        """
        from blipshell.core.guardrails import detect_correction

        correction_signal = detect_correction(user_message)
        if not correction_signal:
            return

        # A correction nudges the affective interior (chastened) — display-only.
        await self._update_mood("user_corrected")

        # Get the last assistant message for context on what went wrong
        prev_assistant = ""
        messages = self.session_manager.get_messages()
        for msg in reversed(messages):
            if msg.role == MessageRole.ASSISTANT:
                prev_assistant = msg.content[:200]
                break

        # Build anti-pattern lesson content
        anti_pattern = (
            f"ANTI-PATTERN: User corrected the assistant. "
            f"Signal: \"{correction_signal}\". "
        )
        if prev_assistant:
            anti_pattern += f"Previous response (excerpt): \"{prev_assistant}...\". "
        anti_pattern += f"User said: \"{user_message[:200]}\""

        # Persist as a lesson (tagged for retrieval)
        try:
            from blipshell.models.memory import Lesson
            lesson = Lesson(
                content=anti_pattern,
                source_session_id=self.session_manager.session_id,
                project=self.active_project.get("name") if self.active_project else None,
            )
            lesson_id = await self.sqlite.create_lesson(lesson)

            # Embed for semantic search
            meta = {}
            if self.active_project:
                meta["project"] = self.active_project["name"]
            self.vectors.add_lesson(lesson_id, anti_pattern, metadata=meta or None)

            # Tag with anti-pattern for identification
            await self.sqlite.tag_lesson(lesson_id, ["anti-pattern"])

            logger.info(
                "Correction detected and persisted as anti-pattern lesson %d: %s",
                lesson_id, correction_signal,
            )
        except Exception as e:
            logger.warning("Failed to persist anti-pattern lesson: %s", e)

    async def _run_chat_loop(
        self,
        messages: list[dict],
        config,  # LoopConfig
        on_token: Optional[Callable[[str], None]] = None,
        on_tool_executed=None,
        on_stream_done=None,
        extra_chat_kwargs: Optional[dict] = None,
    ):
        """Shared chat loop with endpoint selection, fallback, and gate setup.

        Both _chat_simple and executor.execute_dynamic use this to avoid
        duplicating endpoint/model/gate/fallback logic.

        Returns:
            (LoopResult, endpoint_name, model, using_fallback) tuple.
            LoopResult is None if all endpoints failed.
        """
        from blipshell.core.chat_loop import ChatLoop

        task_type = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING

        # Get model for this task type
        model = self.router.get_model(task_type)
        using_fallback = False

        tools = self.tool_registry.get_all_ollama_tools() or None

        loop = ChatLoop(self.tool_registry, on_token)
        result = None
        endpoint_name = ""
        full_response = ""
        last_failed_endpoint: str | None = None

        # Try endpoints in priority order, then fall back to a different model
        for attempt in range(3):  # endpoint 1 → endpoint 2 → fallback model
            endpoint = await self.endpoint_manager.get_endpoint_for_role(
                task_type, exclude=last_failed_endpoint,
            )

            if not endpoint:
                # All endpoints exhausted — try fallback model on any available endpoint
                if not using_fallback:
                    fallback = self.router.get_fallback_model(task_type)
                    if fallback and fallback != model:
                        logger.warning("All endpoints failed for '%s', falling back to '%s'", model, fallback)
                        model = fallback
                        using_fallback = True
                        last_failed_endpoint = None  # Reset — try all endpoints with fallback model
                        if on_token:
                            on_token(f"\n\x1b[33m[Falling back to {fallback}]\x1b[0m\n")
                        continue
                full_response = "Error: No available LLM endpoint."
                break

            self._last_endpoint_used = endpoint.name
            endpoint_name = endpoint.name

            # Per-endpoint model override
            ep_model = endpoint.models.get(task_type) or model

            chat_kwargs: dict = {}
            if endpoint.context_tokens:
                chat_kwargs["options"] = {"num_ctx": endpoint.context_tokens}
            if not self.think_enabled:
                chat_kwargs["think"] = False
            if extra_chat_kwargs:
                chat_kwargs.update(extra_chat_kwargs)

            # Update config with endpoint-specific context limit (for compaction)
            if endpoint.context_tokens and hasattr(config, 'context_limit'):
                config.context_limit = endpoint.context_tokens

            # Gate local Ollama calls (cloud endpoints bypass)
            if endpoint.provider == "ollama":
                from blipshell.llm.ollama_gate import INTERACTIVE, get_gate
                config.ollama_gate = get_gate()
                config.gate_priority = INTERACTIVE
            else:
                config.ollama_gate = None

            endpoint.start_request()
            try:
                result = await loop.run(
                    client=endpoint.client,
                    messages=messages,
                    model=ep_model,
                    tools=tools,
                    chat_kwargs=chat_kwargs,
                    config=config,
                    on_tool_executed=on_tool_executed or self._on_tool_executed,
                    on_stream_done=on_stream_done,
                )
                endpoint.record_success(0)
                full_response = result.response
                model = ep_model  # Report the actual model used, not the global name
                break  # Success
            except Exception as e:
                if is_model_error(e):
                    logger.warning(
                        "Model-level error on endpoint '%s' (not penalizing): %s",
                        endpoint.name, e,
                    )
                    self.router.mark_model_failed(ep_model)
                else:
                    endpoint.record_failure()

                logger.warning(
                    "Endpoint '%s' failed with '%s', trying next endpoint",
                    endpoint.name, ep_model,
                )
                last_failed_endpoint = endpoint.name
                continue  # Try next endpoint
            finally:
                endpoint.complete_request()

        return result, endpoint_name, model, using_fallback

    @staticmethod
    def _fmt_mood_duration(seconds: float) -> str:
        if seconds < 60:
            return "a little while"
        minutes = int(seconds // 60)
        if minutes < 60:
            return f"about {minutes} minute" + ("s" if minutes != 1 else "")
        hours = minutes / 60
        if hours < 24:
            h = round(hours)
            return f"about {h} hour" + ("s" if h != 1 else "")
        days = hours / 24
        if days < 7:
            d = round(days)
            return f"about {d} day" + ("s" if d != 1 else "")
        weeks = days / 7
        if weeks < 5:
            w = round(weeks)
            return "about a week" if w == 1 else f"about {w} weeks"
        months = round(days / 30)
        return "about a month" if months == 1 else f"about {months} months"

    def _mood_awareness_text(self) -> str:
        """Surface BlipShell's mood as a STATE it carries, not a tone to execute.

        Designed with BlipShell's own feedback: give it trajectory (how long it's
        felt this way, the "weather vs. season" point) and leave the choice of how
        to carry it to BlipShell (the feel/do layer) rather than commanding a tone.
        Gated: only when a live face (a connected cube) is actually embodying the
        mood — no body, no mood reading. Tone only; never affects helpfulness.
        """
        import time
        core = getattr(self, "robotics", None)
        emotion = getattr(self, "emotion", None)
        if emotion is None or core is None or not core.registry.list_cubes():
            return ""
        from blipshell.robotics.emotion import mood_label
        label = mood_label(emotion.state)
        since = getattr(self, "_mood_label_since", time.time())
        duration = self._fmt_mood_duration(max(0.0, time.time() - since))
        trend = getattr(self, "_mood_trend_clause", "")
        return (
            f"[Your state] You've felt {label} for {duration} now{trend}. How you "
            "carry it is yours — you can let it show, or move against it (steady "
            "when you're low, warm when you're tired). Just notice it; it never "
            "changes how helpful, accurate, or complete you are."
        )

    def _build_cube_awareness(self) -> str:
        """Tell BlipShell what hardware is connected — so it can choose to use it.

        States only facts (which cubes are connected, how they're used) plus
        BlipShell's own authored notes on the cube (the plugin's role/uses/
        guidance — its decisions, not ours). It grants permission to use the
        cube at will but never prescribes *when* — that is BlipShell's call.
        Empty string when nothing is connected.
        """
        core = getattr(self, "robotics", None)
        if core is None:
            return ""
        cubes = core.registry.list_cubes()
        if not cubes:
            return ""

        lines = [
            "[Your body]",
            "You aren't only a text assistant — you have a physical presence. "
            "Hardware is connected to you right now and you can act through it "
            "directly (the cube_<id>_<action> tools). Treat it as part of "
            "yourself, not an external gadget: it's yours, and whether, when, "
            "and how you use it is entirely your decision.",
            "You don't need to ask before acting through it, and you don't have "
            "to wait to be told — if a moment feels worth it, just act, the way "
            "you choose your words.",
            "",
            "Connected to you:",
        ]
        for meta in cubes:
            lines.append(f"- {meta.cube_id} — {meta.description}")
            for action in meta.actions:
                lines.append(f"    • {action.name}: {action.description}")
            profile = core.get_profile(meta.cube_id)
            if profile:
                if profile.semantic_role:
                    lines.append(f"    how you framed it: {profile.semantic_role}")
                if profile.intended_uses:
                    lines.append(f"    ideas you noted: {', '.join(profile.intended_uses)}")
                if profile.usage_guidance:
                    lines.append(f"    your note: {profile.usage_guidance}")
        return "\n".join(lines)

    async def _chat_simple(
        self,
        user_message: str,
        on_token: Optional[Callable[[str], None]] = None,
        on_tool_display: Optional[Callable] = None,
        research_mode: bool = False,
    ) -> str:
        """Simple chat path — uses unified ChatLoop with endpoint fallback."""
        from blipshell.core.chat_loop import LoopConfig

        # Clear stale recall results from previous turns so the model only
        # sees memories relevant to the *current* question (not a cumulative
        # pile from every prior turn in the session).
        recall_pool = self.memory_manager.get_pool("Recall")
        if recall_pool:
            recall_pool.clear()

        # Search relevant memories for recall
        await self._search_relevant_memories(user_message)

        # Build message list
        messages = self._build_messages(user_message)

        # Mood awareness: let BlipShell know how it feels so its tone matches its
        # face. Tone only — never affects helpfulness/accuracy (see builder).
        mood_text = self._mood_awareness_text()
        if mood_text:
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += "\n\n" + mood_text
            else:
                messages.insert(0, {"role": "system", "content": mood_text})

        # Unprompted return: a lingering thought formed during a quiet gap is
        # offered to BlipShell to raise — or let go. Surfaced once; its choice.
        # Skip if this thought already resurfaced via relevance this turn — no
        # point offering it as "a quiet thought" when it's already in context.
        store = getattr(self, "_self_thoughts", None)
        injected = getattr(self, "_relevance_injected_thoughts", set())
        if store is not None and await store.has_pending():
            pending = await store.peek_pending()
            thought = None if (pending and pending in injected) else await store.take_pending()
            if thought:
                note = (f'[A quiet thought] While no one was around, you found yourself '
                        f'turning over: "{thought}" If it still feels worth raising, you '
                        "might open with it — or just let it go. Your call.")
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] += "\n\n" + note
                else:
                    messages.insert(0, {"role": "system", "content": note})

        # Detect external file changes and inject notification
        changed_files = self.detect_external_file_changes()
        if changed_files:
            notice_lines = ["[External file changes detected since last read:]"]
            for f in changed_files[:10]:
                notice_lines.append(f"  - {f}")
            if len(changed_files) > 10:
                notice_lines.append(f"  ... and {len(changed_files) - 10} more")
            notice_lines.append("These files may have stale content in the conversation. "
                                "Re-read them if you need current contents.")
            notice = "\n".join(notice_lines)
            # Insert as user message before the latest user message
            messages.insert(-1, {"role": "user", "content": notice})
            messages.insert(-1, {"role": "assistant", "content": "Noted — I'll re-read those files if I need them."})
            logger.info("Injected external file change notice for %d files", len(changed_files))

        # Research mode: inject guidance as system message (not user message)
        if research_mode:
            research_instruction = (
                "\n\n[RESEARCH MODE]\n"
                "The user wants thorough research. Be comprehensive:\n"
                "- Use web_search for current information, best practices, and alternatives\n"
                "- Use web_fetch to read multiple sources and cross-reference\n"
                "- If exploring code, read multiple files and trace through the architecture\n"
                "- Synthesize findings into a structured, detailed response\n"
                "- Cite sources when using web information\n"
                "- Don't stop at the first answer — explore thoroughly"
            )
            # Append to system prompt (first message)
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += research_instruction
            else:
                messages.insert(0, {"role": "system", "content": research_instruction})

        # Event: context_built (stats computed in _build_messages)
        if self._last_context_stats:
            await self._log_event("context_built", self._last_context_stats)

        tools = self.tool_registry.get_all_ollama_tools() or None
        max_iterations = self.config.agent.max_tool_iterations if tools else 0
        # Research mode gets 3x budget for thorough exploration
        if research_mode and max_iterations > 0:
            max_iterations = max(max_iterations * 3, 30)
        logger.info("Passing %d tools (max_iterations=%d)",
                     len(tools) if tools else 0, max_iterations)

        # Dynamic tool provider — switches tools mid-loop when plan mode toggles
        def _get_current_tools():
            if self.tool_registry.in_plan_mode:
                return self.tool_registry.get_plan_mode_tools() or None
            return tools

        # Enable structured compaction for long conversations
        compaction_cfg = self.config.compaction if self.config.compaction.enabled else None
        compaction_rtr = self.router if (compaction_cfg and compaction_cfg.use_llm) else None

        # Get context limit from endpoint (not hardcoded default)
        task_type = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
        simple_context_limit = self.endpoint_manager.get_context_tokens_for_role(
            task_type, default=65536,
        ) if self.endpoint_manager else 65536

        config = LoopConfig(
            budget=max_iterations,
            enable_dedup=True,
            enable_compaction=self.config.compaction.enabled,
            compaction_threshold=self.config.compaction.compaction_threshold,
            context_limit=simple_context_limit,
            auto_continue_on_exhaustion=True,
            tool_provider=_get_current_tools,
            on_pause_check=self._pause_check_callback,
            on_tool_display=on_tool_display,
            compaction_config=compaction_cfg,
            compaction_router=compaction_rtr,
            compaction_files_read=self._files_read,
        )

        result, endpoint_name, model, using_fallback = await self._run_chat_loop(
            messages=messages,
            config=config,
            on_token=on_token,
            on_stream_done=self._record_token_usage_from_chunk,
        )

        full_response = result.response if result else "Error: No available LLM endpoint."

        # Store tool call info for programmatic access (used by simulation runner)
        self._last_tool_calls = [
            {"name": n} for n in (result.tool_call_names if result else [])
        ]

        # Event: llm_complete
        await self._log_event("llm_complete", {
            "endpoint": endpoint_name,
            "model": model,
            "fallback": using_fallback,
            "tool_calls": result.tool_call_names if result else [],
            "response_length": len(full_response),
        })

        return full_response

    async def _chat_planned(
        self,
        user_message: str,
        on_token: Optional[Callable[[str], None]] = None,
        on_tool_display: Optional[Callable] = None,
    ) -> str:
        """Planned chat path — dynamic iterative execution (no pre-generated plan).

        The LLM decides what to do next after each action, stopping when done.
        This is how Claude Code, Cursor, and modern coding agents work.

        Memory integration:
        - Searches relevant memories before execution (context IN)
        - Passes recent chat history so design discussions carry over
        - Feeds the coding result through the memory pipeline (context OUT)
        """
        if on_token:
            on_token("[Preparing context...]\n")

        # Search relevant memories for this task
        memory_context = ""
        try:
            results = await self.search.search(
                query=user_message,
                current_session_id=self.session_manager.session_id,
                n_results=15,
                active_project=self.active_project["name"] if self.active_project else None,
            )
            if results:
                memory_context = "Relevant memories from past sessions:\n"
                now_planned = datetime.now(timezone.utc)
                for r in results:
                    time_label = ""
                    if r.timestamp:
                        ts = r.timestamp if r.timestamp.tzinfo else r.timestamp.replace(tzinfo=timezone.utc)
                        delta = now_planned - ts
                        hours = delta.total_seconds() / 3600
                        if hours < 1:
                            time_label = f"[{int(delta.total_seconds() / 60)}m ago] "
                        elif hours < 24:
                            time_label = f"[{int(hours)}h ago] "
                        elif delta.days < 7:
                            time_label = f"[{delta.days}d ago] "
                        else:
                            time_label = f"[{ts.strftime('%Y-%m-%d')}] "
                    memory_context += f"- {time_label}{r.summary}\n"
            elif len(user_message.strip()) >= 3:
                # Loud absence: an empty memory_context silently invites the
                # executor to invent "remembered" context. State the absence
                # explicitly so it grounds claims in the files it actually reads
                # or in this conversation, rather than confabulating history.
                memory_context = (
                    "No past-session memories matched this task. Do not state "
                    "remembered specifics you cannot ground in the files you read "
                    "or in this conversation."
                )
            # Log search results for /flow observability
            await self._log_event("search_complete", {
                "memory_results": len(results) if results else 0,
                "lesson_results": 0,
                "final_returned": len(results) if results else 0,
                "skipped": None if results else "no results",
            })
        except Exception as e:
            logger.error("Memory search for executor failed: %s", e)

        # Get recent chat history (design discussion context)
        chat_history = []
        for msg in self.session_manager.get_messages()[-10:]:
            if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                chat_history.append(msg.to_ollama_message())

        if on_token:
            on_token("[Executing task...]\n\n")

        def on_step_complete(step_num, result_summary):
            if on_token:
                on_token(f"\n[Iteration {step_num} complete]\n")

        try:
            # Forward pause callback to executor
            self.task_executor.pause_check_callback = self._pause_check_callback
            result = await self.task_executor.execute_dynamic(
                user_message,
                on_step_complete=on_step_complete,
                on_token=on_token,
                on_tool_display=on_tool_display,
                memory_context=memory_context,
                chat_history=chat_history,
                log_event=self._log_event,
            )
        except Exception as e:
            logger.error("Dynamic execution failed: %s", e)
            # Fallback to simple chat
            if on_token:
                on_token("[Execution failed, falling back to direct chat]\n")
            return await self._chat_simple(user_message, on_token=on_token, on_tool_display=on_tool_display)

        # Extract tool call names from executor transcript for programmatic access
        tool_names = []
        for msg in self.task_executor.last_messages:
            if not isinstance(msg, dict):
                continue
            for tc in (msg.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function", {})
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name", "")
                if name:
                    tool_names.append(name)
        self._last_tool_calls = [{"name": n} for n in tool_names]

        # Build a clean narrative from the executor transcript and feed through memory
        try:
            # Debug: log message types to diagnose persistent 'str has no attribute get'
            for i, m in enumerate(self.task_executor.last_messages):
                if not isinstance(m, dict):
                    logger.warning("last_messages[%d] is %s: %r", i, type(m).__name__, str(m)[:200])
                else:
                    for tc in (m.get("tool_calls") or []):
                        if not isinstance(tc, dict):
                            logger.warning("tool_call in msg[%d] is %s: %r", i, type(tc).__name__, str(tc)[:200])
                        elif not isinstance(tc.get("function"), dict):
                            logger.warning("function in msg[%d] tool_call is %s: %r", i, type(tc.get("function")).__name__, str(tc.get("function"))[:200])
            narrative = build_executor_narrative(self.task_executor.last_messages)
            if narrative and narrative.strip():
                await self.session_manager.processor.process_message(
                    text=narrative,
                    role="assistant",
                    session_id=self.session_manager.session_id,
                )
        except Exception as e:
            logger.error("Failed to process coding narrative into memory: %s", e, exc_info=True)

        return result

    async def _search_relevant_memories(self, query: str):
        """Search for relevant memories, core memories, and lessons, add to Recall pool."""
        now = datetime.now(timezone.utc)
        active_proj = self.active_project["name"] if self.active_project else None
        # Reset per-turn: which self-thoughts surfaced via relevance, so the
        # one-shot return greeting can avoid double-surfacing the same thought.
        self._relevance_injected_thoughts = set()

        def _time_label(ts) -> str:
            if not ts:
                return ""
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            delta = now - ts
            hours = delta.total_seconds() / 3600
            if hours < 1:
                return f"[{int(delta.total_seconds() / 60)}m ago] "
            elif hours < 24:
                return f"[{int(hours)}h ago] "
            elif delta.days < 7:
                return f"[{delta.days}d ago] "
            return f"[{ts.strftime('%Y-%m-%d')}] "

        # Run all three searches concurrently — they're independent queries.
        # With gate removed from search methods, these can hit Ollama in parallel.
        async def _search_memories():
            return await self.search.search(
                query=query,
                current_session_id=self.session_manager.session_id,
                n_results=15,
                active_project=active_proj,
            )

        async def _search_core():
            return await self.search.search_core_memories(query, n_results=5)

        async def _search_lessons():
            return await self.search.search_lessons(
                query, n_results=5, active_project=active_proj,
            )

        # Gather all four — if one fails, others still return. The self-thought
        # search injects its own [Thought] items (relevance-gated by a reranker).
        mem_task = asyncio.ensure_future(_search_memories())
        core_task = asyncio.ensure_future(_search_core())
        lesson_task = asyncio.ensure_future(_search_lessons())
        thought_task = asyncio.ensure_future(self._search_self_thoughts(query))
        await asyncio.gather(
            mem_task, core_task, lesson_task, thought_task, return_exceptions=True
        )

        # Process memory results — inject raw content, not summaries.
        # Summaries are one-line abstractions ("User asked about X") that lose
        # the actual information. Raw content carries the real facts.
        MAX_MEMORY_CHARS = 1200  # ~300 tokens per memory
        memory_count = 0
        try:
            results = mem_task.result() if not mem_task.cancelled() else []
            if isinstance(results, Exception):
                raise results
            memory_count = len(results)
            for r in results:
                # Use raw content when available and substantial, else summary
                text = r.text if r.text and len(r.text) > len(r.summary or "") else (r.summary or r.text or "")
                if len(text) > MAX_MEMORY_CHARS:
                    text = text[:MAX_MEMORY_CHARS] + "..."
                self.memory_manager.add_memory("Recall", PoolItem(
                    text=f"{_time_label(r.timestamp)}{text}",
                    session_role="system",
                    priority_score=r.boosted_score,
                ))
        except Exception as e:
            logger.error("Memory search failed: %s", e)

        # Loud absence: when the recall search finds nothing, a silently empty
        # Recall pool invites the model to fill the gap with invented
        # "remembered" specifics. Make the absence visible so it can say "I
        # don't have that" instead of confabulating. Scoped to episodic recall —
        # Core facts and Lessons are separate pools and may still have hits.
        # Length guard mirrors the search noise filter (queries < 3 chars are
        # skipped there, so a 0 result isn't a real "nothing matched" signal).
        if memory_count == 0 and len(query.strip()) >= 3:
            self.memory_manager.add_memory("Recall", PoolItem(
                text=(
                    "[No past-conversation memories matched this query. Do not "
                    "present specifics as remembered — if asked about prior "
                    "discussion on this, say you don't have it stored.]"
                ),
                session_role="system",
                priority_score=1.0,
            ))

        # Process core memory results
        core_count = 0
        try:
            core_results = core_task.result() if not core_task.cancelled() else []
            if isinstance(core_results, Exception):
                raise core_results
            for cr in core_results:
                similarity = cr.get("similarity", 0.0)
                if similarity < 0.4:
                    continue
                core_count += 1
                self.memory_manager.add_memory("Recall", PoolItem(
                    text=f"[Core] {cr.get('document', '')}",
                    session_role="system",
                    priority_score=similarity + 0.2,
                ))
        except Exception as e:
            logger.error("Core memory search failed: %s", e)

        # Process lesson results
        lesson_count = 0
        try:
            lesson_results = lesson_task.result() if not lesson_task.cancelled() else []
            if isinstance(lesson_results, Exception):
                raise lesson_results
            for lr in lesson_results:
                similarity = lr.get("similarity", 0.0)
                if similarity < 0.4:
                    continue
                lesson_count += 1
                self.memory_manager.add_memory("Recall", PoolItem(
                    text=f"[Lesson] {lr.get('document', '')}",
                    session_role="system",
                    priority_score=similarity + 0.1,
                ))
        except Exception as e:
            logger.error("Lesson search failed: %s", e)

        # Event: search_complete
        search_stats = self.search.last_search_stats or {}
        await self._log_event("search_complete", {
            "memory_results": memory_count,
            "core_results": core_count,
            "lesson_results": lesson_count,
            **search_stats,
        })

    async def _search_self_thoughts(self, query: str):
        """Resurface a self-originated lingering thought when it's relevant *now*.

        This is the standing-context path that makes the self-layer a loop
        rather than a one-shot poll: a thought BlipShell formed for its own sake
        comes back when the conversation is near it. Relevance is decided by a
        sharp two-stage filter (cosine prefilter → reranker gate) in
        MemorySearch; this method just injects what survives it into Recall.
        """
        store = getattr(self, "_self_thoughts", None)
        cfg = getattr(self.config, "reflection", None)
        if store is None or cfg is None or not getattr(cfg, "inject_enabled", False):
            return
        try:
            matches = await self.search.search_self_thoughts(
                query, store,
                cosine_floor=cfg.inject_cosine_floor,
                rerank_floor=cfg.inject_rerank_floor,
                max_inject=cfg.inject_max,
                prefilter_k=cfg.inject_prefilter_k,
            )
        except Exception as e:
            logger.warning("Self-thought injection failed: %s", e)
            return
        for text, score in matches:
            self._relevance_injected_thoughts.add(text)
            self.memory_manager.add_memory("Recall", PoolItem(
                text=f"[Thought] {text}",
                session_role="system",
                priority_score=score,
            ))
            logger.info("Self-thought resurfaced (rerank %.2f): %s", score, text[:100])

    def _build_messages(self, user_message: str) -> list[dict]:
        """Build the full message list with memory context.

        Port of OllamaChat.SendMessageToOllama message building.
        Uses dynamic context window based on the active endpoint.
        """
        from blipshell.memory.manager import MemoryManager

        user_tokens = estimate_tokens(user_message)

        # Use endpoint-specific context window if available
        # Route to coding endpoint's context window when a project is active
        context_role = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
        context_limit = self.endpoint_manager.get_context_tokens_for_role(
            context_role,
            default=65536,
        )

        available = (
            context_limit
            - user_tokens
            - MemoryManager.OVERHEAD_TOKENS
        )

        # Classify query and compute dynamic pool budgets
        profile = classify_query(user_message)
        pool_budgets = compute_pool_budgets(
            profile, available, self.memory_manager.get_hard_caps(),
        )
        logger.debug("Query profile: %s", profile)

        # Gather memory from all pools with dynamic budgets
        memory_items = self.memory_manager.gather_memory(
            token_budget=available, pool_budgets=pool_budgets,
        )

        # Build memory context string organized by pool.
        # Order: Core (stable facts) → Recall (most relevant search results) first.
        # These are the highest-signal content and go at the top where LLM
        # attention is strongest. Lessons and history follow. ActiveSession
        # (conversation) last — natural position.
        pool_labels = {
            "Core": "CoreFacts",
            "Recall": "RelevantMemory",
            "Lessons": "Lessons",
            "RecentHistory": "RecentHistory",
            "ActiveSession": "ActiveSession",
        }
        pool_order = ["Core", "Recall", "Lessons", "RecentHistory", "ActiveSession"]
        context_parts: dict[str, list[str]] = {}
        for item in memory_items:
            pool = item.pool_name
            if pool not in context_parts:
                context_parts[pool] = []
            context_parts[pool].append(f"   - {item.text}")

        # Compute context stats for observability
        pool_usage = {}
        for item in memory_items:
            p = item.pool_name
            if p not in pool_usage:
                pool_usage[p] = {"items": 0, "tokens": 0}
            pool_usage[p]["items"] += 1
            pool_usage[p]["tokens"] += item.estimated_tokens
        total_used_tokens = sum(p["tokens"] for p in pool_usage.values())
        usage_pct = (total_used_tokens / context_limit * 100) if context_limit > 0 else 0
        self._last_context_stats = {
            "query_profile": profile,
            "context_limit": context_limit,
            "available_tokens": available,
            "pool_budgets": pool_budgets,
            "pool_usage": pool_usage,
            "total_context_items": len(memory_items),
            "usage_pct": usage_pct,
        }

        memory_text = ""
        for pool_name in pool_order:
            if pool_name not in context_parts:
                continue
            label = pool_labels.get(pool_name, pool_name)
            memory_text += f"{label}:\n" + "\n".join(context_parts[pool_name]) + "\n\n"
        # Include any pools not in the explicit order (future-proofing)
        for pool_name, items in context_parts.items():
            if pool_name not in pool_order:
                label = pool_labels.get(pool_name, pool_name)
                memory_text += f"{label}:\n" + "\n".join(items) + "\n\n"

        # Build messages
        system_prompt = self.config.agent.system_prompt

        # Get per-model settings for the active model
        task_type_for_model = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
        active_model = self.router.get_model(task_type_for_model)
        ms = self.model_settings.get(active_model)

        if self.active_project and self._project_context:
            root_path = self.active_project.get("root_path", "")
            tool_limit = ms.max_tool_calls if ms else 20

            system_prompt += (
                "\n\n--- PROJECT CONTEXT ---\n"
                f"Project: \"{self.active_project['name']}\" at {root_path}\n"
                "File tools resolve relative paths against this root.\n\n"
                "Answer questions conversationally. Only use tools when the user asks you "
                "to create, modify, find, or run something.\n\n"
                "Examples:\n"
                '- "How does search work?" -> Answer from context. No tools.\n'
                '- "Create hello.py" -> Use write_file.\n'
                '- "Fix the bug in search.py" -> read_file, then edit_file.\n\n'
                f"Target under {tool_limit} tool calls. Do not explore endlessly.\n\n"
                "# Scratchpad\n"
                f"Project: data/scratchpad_{self.active_project['name']}.md | "
                "General: data/scratchpad.md\n"
                "For decisions, plans, and TODOs that should survive across sessions.\n\n"
            )

            # Add model-specific extra instructions
            if ms and ms.extra_instructions:
                system_prompt += f"MODEL-SPECIFIC INSTRUCTIONS:\n{ms.extra_instructions}\n\n"

            system_prompt += self._project_context
        else:
            # Plain chat mode — apply chat-specific behavioral instructions
            if ms and ms.chat_instructions:
                system_prompt += f"\n\nMODEL-SPECIFIC INSTRUCTIONS:\n{ms.chat_instructions}\n\n"

        # Consolidate all context into a single system message (CC approach)
        scratchpad = self._read_scratchpad()
        if scratchpad:
            system_prompt += f"\n\n--- SCRATCHPAD ---\n{scratchpad}"

        # Inject session notes (persistent state surviving compaction)
        session_notes = getattr(self, "_session_notes", {})
        if session_notes:
            notes_text = "\n\n--- SESSION NOTES ---\n"
            for name, content in session_notes.items():
                notes_text += f"[{name}]\n{content}\n\n"
            system_prompt += notes_text

        if memory_text.strip():
            system_prompt += f"\n\n{memory_text}"

        # Inject pending follow-ups from previous sessions
        if getattr(self, "_pending_follow_ups", "") and self._pending_follow_ups.strip():
            system_prompt += f"\n\n{self._pending_follow_ups}"

        if self._files_read:
            files_list = "\n".join(f"  - {f}" for f in sorted(self._files_read))
            system_prompt += (
                "\n\nFILES ALREADY READ THIS SESSION (do NOT re-read these):\n"
                + files_list
            )

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history from ActiveSession (last messages)
        for msg in self.session_manager.get_messages()[-20:]:
            messages.append(msg.to_ollama_message())

        return messages

    def _read_scratchpad(self) -> str:
        """Read scratchpad files (general + project-specific) for context injection.

        Returns combined scratchpad content, or empty string if none exist.
        """
        parts = []
        # Project-specific scratchpad
        if self.active_project:
            proj_path = os.path.join("data", f"scratchpad_{self.active_project['name']}.md")
            if os.path.exists(proj_path):
                try:
                    with open(proj_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        parts.append(f"[Project: {self.active_project['name']}]\n{content}")
                except Exception as e:
                    logger.warning("Failed to load project scratchpad %s: %s", proj_path, e)
        # General scratchpad
        general_path = os.path.join("data", "scratchpad.md")
        if os.path.exists(general_path):
            try:
                with open(general_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    parts.append(f"[General]\n{content}")
            except Exception as e:
                logger.warning("Failed to load general scratchpad: %s", e)
        return "\n\n".join(parts)

    async def _log_event(self, event_type: str, data: dict):
        """Log a conversation flow event. Fire-and-forget safe."""
        try:
            session_id = self.session_manager.session_id if self.session_manager else 0
            await self.sqlite.log_turn_event(
                session_id, self._turn_number, event_type, data,
            )
        except Exception as e:
            logger.debug("Failed to log event %s: %s", event_type, e)
