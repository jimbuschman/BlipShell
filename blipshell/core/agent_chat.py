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

        # Track user activity for nightly scheduler
        import time
        self._last_user_activity = time.time()

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

        # Add assistant response to session (skip empty — prevents cascade of
        # blank responses where an empty assistant message confuses the model)
        if response and response.strip():
            self.session_manager.add_message(MessageRole.ASSISTANT, response)

        # Background: dump to memory periodically (tracked for clean shutdown)
        task = asyncio.create_task(self._background_memory_processing())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

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
            self.chroma.add_lesson(lesson_id, anti_pattern, metadata=meta or None)

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

        config = LoopConfig(
            budget=max_iterations,
            enable_dedup=True,
            enable_compaction=self.config.compaction.enabled,
            compaction_threshold=self.config.compaction.compaction_threshold,
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
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                name = fn.get("name", "")
                if name:
                    tool_names.append(name)
        self._last_tool_calls = [{"name": n} for n in tool_names]

        # Build a clean narrative from the executor transcript and feed through memory
        try:
            narrative = build_executor_narrative(self.task_executor.last_messages)
            if narrative and narrative.strip():
                await self.session_manager.processor.process_message(
                    text=narrative,
                    role="assistant",
                    session_id=self.session_manager.session_id,
                )
        except Exception as e:
            logger.error("Failed to process coding narrative into memory: %s", e)

        return result

    async def _search_relevant_memories(self, query: str):
        """Search for relevant memories and lessons, add to Recall pool."""
        # Search conversation memories
        memory_count = 0
        try:
            active_proj = self.active_project["name"] if self.active_project else None
            results = await self.search.search(
                query=query,
                current_session_id=self.session_manager.session_id,
                n_results=15,
                active_project=active_proj,
            )
            memory_count = len(results)
            now = datetime.now(timezone.utc)
            for r in results:
                # Include relative timestamp so LLM can answer temporal queries
                time_label = ""
                if r.timestamp:
                    ts = r.timestamp if r.timestamp.tzinfo else r.timestamp.replace(tzinfo=timezone.utc)
                    delta = now - ts
                    hours = delta.total_seconds() / 3600
                    if hours < 1:
                        time_label = f"[{int(delta.total_seconds() / 60)}m ago] "
                    elif hours < 24:
                        time_label = f"[{int(hours)}h ago] "
                    elif delta.days < 7:
                        time_label = f"[{delta.days}d ago] "
                    else:
                        time_label = f"[{ts.strftime('%Y-%m-%d')}] "
                self.memory_manager.add_memory("Recall", PoolItem(
                    text=f"{time_label}{r.summary}",
                    session_role="system",
                    priority_score=r.boosted_score,
                ))
        except Exception as e:
            logger.error("Memory search failed: %s", e)

        # Search lessons semantically (closes the lessons loop)
        lesson_count = 0
        try:
            active_proj = self.active_project["name"] if self.active_project else None
            lesson_results = await self.search.search_lessons(
                query, n_results=5, active_project=active_proj,
            )
            for lr in lesson_results:
                similarity = lr.get("similarity", 0.0)
                if similarity < 0.4:
                    continue
                lesson_count += 1
                self.memory_manager.add_memory("Recall", PoolItem(
                    text=lr.get("document", ""),
                    session_role="system2",  # labeled as "RelevantLessons" in context
                    priority_score=similarity + 0.1,  # slight boost for lessons
                ))
        except Exception as e:
            logger.error("Lesson search failed: %s", e)

        # Event: search_complete
        search_stats = self.search.last_search_stats or {}
        await self._log_event("search_complete", {
            "memory_results": memory_count,
            "lesson_results": lesson_count,
            **search_stats,
        })

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

        # Build memory context string organized by pool
        # Order: Core first (stable facts, LLM attends to start), history in
        # the middle (lowest attention), Recall last (most relevant, LLM
        # attends to end) — mitigates lost-in-the-middle effect.
        pool_labels = {
            "Core": "CoreFoundation",
            "Lessons": "RelevantLessons",
            "Recall": "RelevantMemory",
            "RecentHistory": "RecentHistory",
            "Buffer": "RecentHistory",
            "ActiveSession": "ActiveSession",
        }
        pool_order = ["Core", "Lessons", "RecentHistory", "Buffer", "ActiveSession", "Recall"]
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
            tool_limit = ms.max_tool_calls

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
            if ms.extra_instructions:
                system_prompt += f"MODEL-SPECIFIC INSTRUCTIONS:\n{ms.extra_instructions}\n\n"

            system_prompt += self._project_context

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
