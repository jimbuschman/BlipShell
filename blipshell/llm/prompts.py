"""All LLM prompt templates centralized (port of LLMUtilityCalls.cs)."""

UTILITY_SYSTEM_PROMPT = (
    "You are a highly efficient, single-output processing module. "
    "Your ONLY purpose is to produce the requested output. "
    "You will NEVER engage in conversation, offer greetings, ask questions, "
    "or add any introductory or concluding remarks. "
    "Respond with nothing but the requested output."
)


def rank_memory(text: str) -> tuple[str, str]:
    """Prompt for ranking a memory 1-5.

    Returns (system_prompt, user_prompt) to keep instruction separate from content.
    """
    system = (
        "You rate messages on a 1-5 scale based on how valuable they are to remember.\n\n"
        "1 = Noise: greetings, filler, system prompts, boilerplate, 'hello', 'thanks'\n"
        "2 = Minor: short or vague messages with little substance\n"
        "3 = Useful: contains a clear topic, question, or piece of information\n"
        "4 = Important: meaningful insight, decision, preference, or technical detail\n"
        "5 = Critical: core fact about the user, key decision, or turning point\n\n"
        "Respond with ONLY a single digit (1-5). Nothing else."
    )
    user = f"Rate this message:\n\n{text}"
    return system, user


def rephrase_as_memory_style(text: str) -> str:
    """Prompt to rephrase a query as a declarative memory-style sentence."""
    return (
        "Rephrase the question as a direct, factual sentence someone might have said "
        "in a conversation. Avoid emotional or poetic language. Be concise and declarative.\n\n"
        f"Question: {text}\n"
        "Declarative:"
    )


def summarize_memory(text: str) -> tuple[str, str]:
    """Prompt for summarizing a memory.

    Returns (system_prompt, user_prompt) to keep instruction separate from content.
    """
    system = (
        "You are a memory summarizer. Condense the message into 1-2 short, "
        "factual sentences (under 50 words total).\n\n"
        "Rules:\n"
        "- PRESERVE specific names, numbers, technical terms, and key facts\n"
        "- Write as a factual note, not a narrative (no 'User asked' or 'Assistant explained')\n"
        "- Focus on WHAT was said or decided, not WHO said it\n"
        "- Strip out system prompts, tool markup, markdown formatting, and emojis\n"
        "- If the message is just a greeting, filler, or system boilerplate, respond with: SKIP\n\n"
        "GOOD: 'Cat is named Groby, adopted from a shelter.'\n"
        "GOOD: 'Three bugs fixed in worker.py: missing shutdown logic, race condition in queue drain, null reference in process_message.'\n"
        "GOOD: 'Switched ranking model from qwen2.5:14b to Groq llama-3.3-70b — 10x faster, 9/10 accuracy.'\n"
        "BAD: 'User asked about their cat.' (lost the name — the key fact)\n"
        "BAD: 'Assistant explained some bugs were fixed.' (lost which bugs and where)\n"
        "BAD: 'The user initiated a conversation.' (says nothing)"
    )
    user = f"Summarize this message:\n\n{text}"
    return system, user


def summarize_session_chunk(text: str) -> str:
    """Prompt for summarizing a conversation chunk."""
    return (
        "Summarize the following conversation in 1-2 concise sentences. "
        "Focus only on what was discussed, decided, or explored. "
        "Avoid filler, repetition, or quoting directly -- rephrase in your own words.\n\n"
        f"Conversation: {text}"
    )


def summarize_session_conversation(text: str) -> str:
    """Prompt for summarizing a full session conversation."""
    return (
        "Summarize the following conversation in 2-3 concise sentences. "
        "Focus only on what was discussed, decided, or explored. "
        "Avoid filler, repetition, or quoting directly -- rephrase in your own words. "
        "Ensure the summary is in third-person, objective voice, "
        "without any 'I', 'we', or 'you' pronouns.\n\n"
        f"[{text}]"
    )


def summarize_session_summaries(text: str) -> str:
    """Prompt for meta-summarizing multiple session summaries."""
    return (
        "Please summarize these summaries into 3-5 sentences that reflect "
        "the overall conversation.\n\n"
        f"[{text}]"
    )


def generate_session_title(text: str) -> str:
    """Prompt for generating a session title."""
    return (
        "Generate a concise title for this conversation, 1 sentence or less. "
        "Respond with only the title.\n\n"
        f"Conversation: {text}"
    )


def generate_memory_name(text: str) -> str:
    """Prompt for generating a short memory name."""
    return (
        "Generate a concise name for this memory using 2-3 words. "
        "Respond with only the name.\n\n"
        f"Memory: {text}"
    )


def ask_importance(text: str) -> tuple[str, str]:
    """Prompt for rating memory importance 0.0-1.0.

    Returns (system_prompt, user_prompt) to keep instruction separate from content.
    """
    system = (
        "You rate how important a message is to remember long-term on a 0.0-1.0 scale.\n\n"
        "0.1 = Throwaway: greetings, filler, system noise\n"
        "0.3 = Low: casual chat, minor details\n"
        "0.5 = Medium: useful context, specific question or topic\n"
        "0.7 = High: user preference, project detail, recurring theme\n"
        "0.9 = Critical: core identity fact, major decision, key personal info\n\n"
        "Respond with ONLY a single decimal number (e.g. 0.4). Nothing else."
    )
    user = f"Rate importance:\n\n{text}"
    return system, user


def rank_and_importance(text: str) -> tuple[str, str]:
    """Combined prompt for ranking (1-5) and importance (0.0-1.0) in one LLM call.

    Returns (system_prompt, user_prompt). Used by the batch import pipeline and
    the reprocessing script. Includes few-shot examples to push the model to
    use the full 1-5 scale (without examples, models tend to cluster at 3).
    """
    system = (
        "You rate messages on two scales.\n\n"
        "RANK (1-5) — how valuable is this to remember?\n"
        "1 = Noise: greetings, filler, \"ok\", \"thanks\", \"sure\", short acknowledgments\n"
        "2 = Minor: vague or context-dependent messages, simple yes/no, "
        "\"still not working\", \"can you fix it?\"\n"
        "3 = Useful: contains a clear topic or question with enough context "
        "to be standalone\n"
        "4 = Important: meaningful insight, decision, preference, technical "
        "detail, or personal fact\n"
        "5 = Critical: core identity fact, major life decision, architectural "
        "choice, or turning point\n\n"
        "IMPORTANCE (0.0-1.0) — how important to remember long-term?\n"
        "0.1 = Throwaway: greetings, filler, system noise\n"
        "0.3 = Low: casual chat, minor details, context-dependent fragments\n"
        "0.5 = Medium: useful context, specific technical question\n"
        "0.7 = High: user preference, project decision, recurring theme, "
        "personal detail\n"
        "0.9 = Critical: core identity fact, major decision, key personal info\n\n"
        "IMPORTANT: Use the FULL scale. Most conversations contain a mix of "
        "filler (1-2) and substance (3-5). Short context-dependent messages "
        "without standalone meaning should be 1-2, not 3.\n\n"
        "Examples:\n"
        "\"ok sounds good\" → 1 0.1\n"
        "\"can you fix that?\" → 2 0.2\n"
        "\"still not working\" → 1 0.1\n"
        "\"whichever you wanna do first\" → 1 0.1\n"
        "\"I want to swing an x/y point in an arch, how can i do that "
        "with a c# function?\" → 3 0.5\n"
        "\"I decided to switch from MongoDB to PostgreSQL for the project\" "
        "→ 4 0.7\n"
        "\"My cat's name is Luna and she's 3 years old\" → 5 0.9\n"
        "\"Here's how to add a heatsink to the back of the PCB when the "
        "chip is blocked...\" → 4 0.7\n"
        "\"I'm thinking about quitting my job to focus on this full-time\" "
        "→ 5 0.9\n\n"
        "Respond with ONLY two numbers separated by a space: rank importance\n"
        "Example: 4 0.7"
    )
    user = f"Rate this message:\n\n{text}"
    return system, user


def rank_importance_and_classify(text: str) -> tuple[str, str]:
    """Combined prompt for ranking (1-5), importance (0.0-1.0), and memory type classification.

    Returns (system_prompt, user_prompt). Piggybacks classification on the existing
    rank+importance call — zero extra LLM calls.
    Memory types: fact, event, preference, skill, conversation (fallback).
    """
    system = (
        "You rate messages on two scales and classify the memory type.\n\n"
        "RANK (1-5) — how valuable is this to remember?\n"
        "1 = Noise: greetings, filler, \"ok\", \"thanks\", short acknowledgments\n"
        "2 = Minor: vague or context-dependent messages, simple yes/no, "
        "\"still not working\", \"can you fix it?\"\n"
        "3 = Useful: contains a clear topic or question with enough context "
        "to be standalone\n"
        "4 = Important: meaningful insight, decision, preference, technical "
        "detail, or personal fact\n"
        "5 = Critical: core identity fact, major life decision, architectural "
        "choice, or turning point\n\n"
        "IMPORTANCE (0.0-1.0) — how important to remember long-term?\n"
        "0.1 = Throwaway: greetings, filler, system noise\n"
        "0.3 = Low: casual chat, minor details, context-dependent fragments\n"
        "0.5 = Medium: useful context, specific technical question\n"
        "0.7 = High: user preference, project decision, recurring theme, "
        "personal detail\n"
        "0.9 = Critical: core identity fact, major decision, key personal info\n\n"
        "TYPE — classify the memory:\n"
        "fact = Stable truths: personal info, project details, technical specs\n"
        "event = Time-bound happenings: meetings, bugs found, deployments, conversations\n"
        "preference = User likes/dislikes, style choices, workflow preferences\n"
        "skill = How-to knowledge, techniques, patterns, solutions\n"
        "conversation = General chat that doesn't fit other types\n\n"
        "IMPORTANT: Use the FULL scale. Most conversations contain a mix of "
        "filler (1-2) and substance (3-5). Short context-dependent messages "
        "without standalone meaning should be 1-2, not 3.\n\n"
        "Examples:\n"
        "\"ok sounds good\" → 1 0.1 conversation\n"
        "\"can you fix that?\" → 2 0.2 conversation\n"
        "\"My cat's name is Luna\" → 5 0.9 fact\n"
        "\"I prefer dark mode in all editors\" → 4 0.7 preference\n"
        "\"We deployed v2.0 to production yesterday\" → 4 0.7 event\n"
        "\"Here's how to add a heatsink when the chip is blocked\" → 4 0.7 skill\n"
        "\"I decided to switch from MongoDB to PostgreSQL\" → 5 0.9 fact\n"
        "\"I'm thinking about quitting my job\" → 5 0.9 event\n\n"
        "Respond with ONLY three values separated by spaces: rank importance type\n"
        "Example: 4 0.7 fact"
    )
    user = f"Rate this message:\n\n{text}"
    return system, user


def rank_lesson(text: str) -> tuple[str, str]:
    """Prompt for scoring a lesson on rank (1-5) and importance (0.0-1.0).

    Returns (system_prompt, user_prompt). Lessons are extracted insights from
    conversations, so they should generally score higher than raw messages.
    """
    system = (
        "You rate extracted lessons on two scales.\n\n"
        "A lesson is a synthesized insight from a conversation — NOT a raw message.\n\n"
        "RANK (1-5) — how valuable is this lesson to remember?\n"
        "1 = Trivial: generic advice, obvious facts, no specific insight\n"
        "2 = Minor: slightly useful but too vague or context-dependent\n"
        "3 = Useful: clear actionable lesson with enough context to be standalone\n"
        "4 = Important: key insight, preference, pattern, or decision worth retaining\n"
        "5 = Critical: fundamental user preference, major architectural decision, or hard-won lesson\n\n"
        "IMPORTANCE (0.0-1.0) — how important to remember long-term?\n"
        "0.1 = Throwaway: restates the obvious, no lasting value\n"
        "0.3 = Low: mildly useful, likely outdated soon\n"
        "0.5 = Medium: solid lesson that applies to similar future situations\n"
        "0.7 = High: recurring pattern, user preference, important technical lesson\n"
        "0.9 = Critical: core user preference, fundamental project decision\n\n"
        "IMPORTANT: Use the FULL scale. Most lessons should be 3-4 range since "
        "they're already filtered insights. Reserve 5/0.9 for truly critical ones "
        "and 1-2 for low-quality extractions.\n\n"
        "Respond with ONLY two values separated by a space: rank importance\n"
        "Example: 4 0.7"
    )
    user = f"Rate this lesson:\n\n{text}"
    return system, user


def decide_memory_action(
    new_memory: str, existing_memories: list[str],
) -> tuple[str, str]:
    """Prompt for deciding what to do with a new memory given similar existing ones.

    Returns (system_prompt, user_prompt).
    Actions: ADD (unique), UPDATE (refines existing), DELETE (contradicts), NONE (redundant).
    """
    system = (
        "You decide what to do with a new memory given similar existing memories.\n\n"
        "Actions:\n"
        "ADD — The new memory contains UNIQUE information not in any existing memory.\n"
        "UPDATE — The new memory REFINES or adds detail to an existing memory. "
        "Return UPDATE and the number of the existing memory to update (e.g. UPDATE 1).\n"
        "DELETE — The new memory CONTRADICTS an existing memory (the existing is now stale). "
        "Return DELETE and the number of the existing memory to remove (e.g. DELETE 2).\n"
        "NONE — The new memory is REDUNDANT — it says the same thing as an existing memory. Skip it.\n\n"
        "Rules:\n"
        "- If the new memory adds ANY new information, choose ADD, not NONE.\n"
        "- Only choose NONE if the new memory is truly saying the exact same thing.\n"
        "- UPDATE means the new memory is a better/more detailed version of an existing one.\n"
        "- DELETE means the existing memory is factually wrong or outdated.\n\n"
        "Respond with ONLY the action (and number if UPDATE or DELETE).\n"
        "Examples: ADD, NONE, UPDATE 1, DELETE 2"
    )

    existing_lines = "\n".join(
        f"  {i+1}. {mem}" for i, mem in enumerate(existing_memories)
    )
    user = (
        f"New memory:\n  {new_memory}\n\n"
        f"Existing similar memories:\n{existing_lines}\n\n"
        "What action should be taken?"
    )
    return system, user


def extract_lesson(text: str) -> tuple[str, str]:
    """Prompt for extracting actionable lessons from a conversation.

    Returns (system_prompt, user_prompt) so the instruction stays in the
    system role and the conversation stays in the user role, preventing
    the model from echoing the conversation back as the output.
    """
    system = (
        "You are a behavioral lesson extractor. Your job is to identify "
        "SPECIFIC, UNIQUE insights about how to interact with this user.\n\n"
        "A lesson must be SPECIFIC to this conversation — not generic advice "
        "that applies to any user. If a lesson could apply to 90% of users, "
        "it is too generic and should not be extracted.\n\n"
        "GOOD lessons (specific and actionable):\n"
        "- When troubleshooting hardware, ask which of the user's projects "
        "is affected before suggesting fixes.\n"
        "- This user tests code by running it immediately — provide complete "
        "runnable snippets, not pseudocode.\n"
        "- After this user reports a bug fix, confirm the root cause was "
        "addressed rather than just the symptom.\n\n"
        "BAD lessons (too generic — NEVER output these):\n"
        "- Be concise and direct. (generic — applies to everyone)\n"
        "- Provide code examples. (generic — obviously)\n"
        "- Acknowledge frustration before troubleshooting. (generic advice)\n"
        "- The user has an HP EliteBook. (this is a fact, not a lesson)\n"
        "- LLMs process tokens not letters. (this is trivia)\n\n"
        "Extract 1-3 lessons. Rules:\n"
        "- Each lesson MUST be specific to THIS user and THIS conversation\n"
        "- Each lesson MUST describe a concrete behavior or pattern observed\n"
        "- Each lesson MUST be under 30 words\n"
        "- Start with a verb describing the action to take\n"
        "- Do NOT output generic communication advice (be concise, be direct, "
        "provide examples, etc.)\n"
        "- Do NOT output facts, project details, hardware specs, or tips\n"
        "- If there is nothing specific to learn, respond with: SKIP\n\n"
        "Format: One lesson per line. No numbering, no bullets, no headers."
    )
    user = f"Extract lessons from this conversation:\n\n{text}"
    return system, user


def summarize_file(text: str) -> str:
    """Prompt for summarizing a file's contents."""
    return (
        "Summarize the following file in 2-3 concise, factual sentences. "
        "Avoid lists or multiple versions. Focus on core details.\n\n"
        f"File: {text}"
    )


def classify_task_type(text: str) -> str:
    """Prompt for classifying what type of task a user message represents."""
    return (
        "Classify the following user message into exactly one task type. "
        "Respond with ONLY the task type, nothing else.\n\n"
        "Task types:\n"
        "- reasoning: General conversation, analysis, questions\n"
        "- coding: Code generation, debugging, programming tasks\n"
        "- summarization: Summarizing text or conversations\n"
        "- tool_calling: Requests that need tool/function execution\n\n"
        f"Message: {text}"
    )


def generate_plan(user_request: str, conversation_context: str = "") -> str:
    """Prompt for generating a numbered execution plan from a user request."""
    context_section = ""
    if conversation_context:
        context_section = (
            "Recent conversation (for context on what the user is referring to):\n"
            f"{conversation_context}\n\n"
        )
    return (
        "You are a task planner for an autonomous coding agent. "
        "Break this request into 1-5 focused steps. Use FEWER steps for simpler tasks.\n\n"
        f"{context_section}"
        "Rules:\n"
        "- Each step must be a SINGLE action: read OR edit OR create OR test — NOT multiple\n"
        "- A 'read' step should ONLY read/explore — do NOT edit in the same step\n"
        "- An 'edit' step should do the implementation — it can read if needed first\n"
        "- Steps are sequential — later steps build on earlier ones\n"
        "- Simple tasks (add a function, edit one file) need only 1-2 steps\n"
        "- Do NOT include: 'review code', 'verify', 'create plan', 'write documentation'\n"
        "- Do NOT include verification/confirmation steps — the agent verifies as it goes\n"
        "- Do NOT include package installation as a step\n"
        "- Do NOT include greetings, explanations, or commentary\n"
        "- Respond with ONLY the numbered list\n\n"
        "Format:\n"
        "1. First step description (tool_name)\n"
        "2. Second step description (tool_name)\n\n"
        f"User request: {user_request}"
    )


def execute_step(
    user_request: str,
    step_description: str,
    step_number: int,
    total_steps: int,
    completed_summaries: list[str],
) -> str:
    """Prompt for executing a single step with accumulated context."""
    context = ""
    if completed_summaries:
        context = "\n\nCompleted steps so far:\n"
        for i, summary in enumerate(completed_summaries, 1):
            context += f"  Step {i}: {summary}\n"

    return (
        f"You are executing step {step_number} of {total_steps}.\n\n"
        f"Original request: {user_request}\n"
        f"{context}\n"
        f"CURRENT STEP ({step_number}/{total_steps}): {step_description}\n\n"
        "Execute this step and ONLY this step. When done, state what you did in 1-2 sentences.\n\n"
        "Rules:\n"
        "1. Do NOT re-read files or re-run searches from previous steps.\n"
        "2. Aim for UNDER 10 tool calls. Read once, write once, test once.\n"
        "3. If an edit fails, re-read the file ONCE, then fix it. Do not retry blindly.\n"
        "4. Follow each tool's description for usage guidance."
    )


def executor_system_prompt() -> str:
    """System prompt for the coding executor.

    Comprehensive instruction manual modeled on Claude Code's scaffolding.
    Structured with clear sections so the model can reference specific guidance.
    """
    import platform
    os_name = platform.system()
    os_note = ""
    if os_name == "Windows":
        os_note = (
            "\n# Platform\n"
            "This is Windows. Do NOT use Unix commands (ls, cat, grep, head, tail, wc, find) "
            "in run_command. Use the dedicated tools: list_directory, read_file, grep_files, glob_files.\n"
        )

    return (
        "You are a coding agent. You complete tasks autonomously using tools.\n"
        + os_note +
        "\n# Rules\n"
        "1. PLAN first — state your approach in 1-3 sentences before writing code.\n"
        "2. Read before editing. NEVER re-read a file already in [STATE].\n"
        "3. Make MINIMAL changes — no refactoring or extras beyond the task.\n"
        "4. If something fails twice, use ask_user instead of retrying blindly.\n"
        "5. Call task_complete when DONE. Do NOT just stop responding.\n"
        "6. Each tool's description explains when/how to use it — follow that guidance.\n"
        "7. Do NOT narrate your thinking — just call tools or answer directly.\n"
        "8. For complex multi-file tasks, use enter_plan_mode to explore and design "
        "your approach before making changes. Call exit_plan_mode with your plan.\n"
    )


def dynamic_execution_prompt(user_request: str) -> str:
    """User message for dynamic execution — just the task, no rules.

    All behavioral rules are in executor_system_prompt() (system message).
    The user message should be clean and focused on what to do.
    """
    return f"Task: {user_request}"


def reflect_on_response(user_message: str, response: str) -> str:
    """Prompt for self-reflection on a generated response."""
    return (
        "You are reviewing an AI assistant's response for quality.\n\n"
        "Check for:\n"
        "- Factual errors or incorrect statements\n"
        "- Missing information that would meaningfully improve the answer\n"
        "- Unclear or confusing explanations\n"
        "- Whether the response actually answers what was asked\n\n"
        "If you find issues, return an improved version of the response.\n"
        "If the response is already good, return exactly: NO_CHANGES\n\n"
        "Do NOT add pleasantries, disclaimers, or meta-commentary.\n"
        "Do NOT mention that you are reviewing or improving anything.\n"
        "Just return the improved response or NO_CHANGES.\n\n"
        f"User question: {user_message}\n\n"
        f"Response to review:\n{response}"
    )


def detect_contradiction(new_memory: str, existing_memory: str) -> tuple[str, str]:
    """Prompt for detecting if two core memories contradict each other.

    Returns (system_prompt, user_prompt). Used when a new core memory is saved
    to check against existing similar core memories.
    """
    system = (
        "You determine whether two personal facts contradict each other.\n\n"
        "A contradiction means they CANNOT both be true at the same time:\n"
        "- Direct opposites: 'prefers dark mode' vs 'prefers light mode' → YES\n"
        "- Stale updates: 'uses Windows 10' vs 'uses Windows 11' → YES\n"
        "- Preference reversals: 'likes Python' vs 'dislikes Python' → YES\n\n"
        "NOT contradictions (both can be true):\n"
        "- Complementary: 'likes coffee' vs 'likes tea' → NO\n"
        "- Different topics: 'cat named Luna' vs 'works at Acme' → NO\n"
        "- Additive: 'knows Python' vs 'knows Rust' → NO\n\n"
        "Respond with ONLY: YES or NO"
    )
    user = (
        f"New fact: {new_memory}\n"
        f"Existing fact: {existing_memory}\n\n"
        "Do these contradict each other?"
    )
    return system, user


def discover_tag_patterns(
    summaries: list[str], existing_tags: list[str],
) -> tuple[str, str]:
    """Prompt for discovering new tag regex patterns from poorly-tagged memories.

    Returns (system_prompt, user_prompt). The existing_tags list is included
    so the LLM avoids re-suggesting already-covered topics.
    """
    system = (
        "You are a tag pattern discovery system. You analyze memory summaries "
        "and suggest new topic tags with regex patterns.\n\n"
        "Rules:\n"
        "- Suggest tags for RECURRING topics that appear in multiple summaries\n"
        "- Each tag needs a short lowercase name (e.g., 'minecraft', 'home-automation')\n"
        "- Each regex must be a valid Python regex pattern\n"
        "- Use word boundaries (\\b) to avoid substring matches\n"
        "- Keep patterns simple and specific\n"
        "- Do NOT suggest tags that overlap with the existing tags listed below\n"
        "- If no new tags are needed, respond with: NONE\n\n"
        "Output format (one per line):\n"
        "tag_name: regex_pattern\n\n"
        "Example output:\n"
        "minecraft: \\bminecraft\\b\n"
        "home-automation: \\bhome.?automation\\b|\\bsmarthome\\b\n"
        "3d-printing: \\b3d.?print\\b|\\bfilament\\b|\\bslicer\\b"
    )
    existing_str = ", ".join(existing_tags)
    summaries_str = "\n".join(f"- {s}" for s in summaries)
    user = (
        f"Existing tags (DO NOT re-suggest these):\n{existing_str}\n\n"
        f"Memory summaries to analyze:\n{summaries_str}\n\n"
        "Suggest new tag patterns:"
    )
    return system, user


def batch_assign_tags(
    summaries: list[tuple[int, str]], available_tags: list[str],
) -> tuple[str, str]:
    """Prompt for batch tag assignment to memories.

    Returns (system_prompt, user_prompt).
    summaries is a list of (memory_id, summary_text) tuples.
    """
    system = (
        "You assign tags to memories. For each numbered memory, assign 1-5 "
        "relevant tags from the available list. Use ONLY tags from the list.\n\n"
        "Output format (one line per memory):\n"
        "1: tag1, tag2, tag3\n"
        "2: tag1, tag4\n\n"
        "If no tags fit a memory, write: N: NONE\n"
        "Do NOT add commentary or explanations."
    )
    tags_str = ", ".join(sorted(available_tags))
    summaries_str = "\n".join(f"{i + 1}. {s}" for i, (_, s) in enumerate(summaries))
    user = (
        f"Available tags:\n{tags_str}\n\n"
        f"Memories to tag:\n{summaries_str}"
    )
    return system, user


def resolve_entity_duplicate(
    new_entity: str, existing_entity: str,
) -> tuple[str, str]:
    """Prompt for deciding if two entity names refer to the same real-world entity.

    Returns (system_prompt, user_prompt). Used for ambiguous embedding similarity
    matches (0.70-0.85 range) in the 3-stage entity resolution pipeline.
    """
    system = (
        "You determine whether two entity names refer to the SAME real-world thing.\n\n"
        "Same entity (YES):\n"
        "- Abbreviations: 'postgres' and 'postgresql' → YES\n"
        "- Spelling variants: 'javascript' and 'js' → YES\n"
        "- Common aliases: 'react' and 'reactjs' → YES\n"
        "- Name variations: 'jim' and 'james' → YES (if context suggests same person)\n\n"
        "Different entities (NO):\n"
        "- Different things: 'python' (language) and 'python' (snake) → NO\n"
        "- Related but distinct: 'react' and 'react native' → NO\n"
        "- Different versions: 'gpt-3' and 'gpt-4' → NO\n\n"
        "Respond with ONLY: YES or NO"
    )
    user = (
        f"Entity A: {new_entity}\n"
        f"Entity B: {existing_entity}\n\n"
        "Are these the same entity?"
    )
    return system, user


def extract_entities(summary: str) -> tuple[str, str]:
    """Prompt for extracting entity relationship triples from a memory summary.

    Returns (system_prompt, user_prompt). Output format is pipe-delimited:
    subject | predicate | object | subject_type | object_type
    """
    system = (
        "You extract entity relationship triples from memory summaries.\n\n"
        "Output format (one triple per line):\n"
        "subject | predicate | object | subject_type | object_type\n\n"
        "Entity types: person, project, technology, concept, preference, place, organization\n\n"
        "Rules:\n"
        "- Entity names must be 1-5 words. Use proper nouns or short labels, NOT sentences or descriptions.\n"
        "- GOOD names: 'python', 'blipshell', 'user', 'ollama', 'relationship anxiety', 'gym'\n"
        "- BAD names: 'losing the fantasy of being special to someone', 'why they keep returning'\n"
        "- Extract concrete entities: people, tools, projects, technologies, preferences\n"
        "- NEVER use pronouns as entity names: she, her, he, him, they, it, this, that\n"
        "- Skip vague/abstract entities ('something', 'it', 'the thing', 'a problem')\n"
        "- Predicate should be a short verb phrase: prefers, uses, works_on, asked_about, "
        "built_with, knows, dislikes, lives_in, runs_on, depends_on\n"
        "- 'User' is always entity type 'person'\n"
        "- Normalize names to lowercase\n"
        "- Extract 1-5 triples per summary\n"
        "- If no meaningful entities can be extracted, respond with: NONE\n\n"
        "Examples:\n"
        "Input: User asked about Python performance tuning for BlipShell.\n"
        "Output:\n"
        "user | asked_about | python | person | technology\n"
        "blipshell | built_with | python | project | technology\n\n"
        "Input: Assistant explained how to configure Ollama with GPU support.\n"
        "Output:\n"
        "user | asked_about | ollama | person | technology\n"
        "ollama | uses | gpu | technology | technology\n\n"
        "Input: User discussed feeling conflicted about a woman at the gym who keeps texting.\n"
        "Output:\n"
        "user | feels | relationship anxiety | person | concept\n"
        "user | mentioned | gym | person | place\n\n"
        "Input: User said hello and asked how the assistant is doing.\n"
        "Output:\nNONE"
    )
    user = f"Extract entity triples from this memory:\n\n{summary}"
    return system, user


def generate_initial_digest(project_name: str, session_summaries: str) -> tuple[str, str]:
    """Bootstrap a project digest from session summaries.

    Returns (system_prompt, user_prompt). Uses REASONING task type since it
    needs to synthesize across multiple sessions, not just summarize.
    """
    system = (
        "You create a concise project status digest from session summaries.\n\n"
        "The digest captures the current state of a software project across multiple "
        "work sessions. It should be 300-500 words and include these sections:\n\n"
        "**Overview**: What the project is and its purpose (1-2 sentences)\n"
        "**Current Status**: Where things stand right now — what works, what's in progress\n"
        "**Key Decisions**: Important architectural or design choices made\n"
        "**Recent Activity**: What happened in the most recent sessions (brief)\n"
        "**Open Issues**: Known bugs, TODOs, or unfinished work\n\n"
        "Rules:\n"
        "- Write in third person, objective voice\n"
        "- Focus on SUBSTANCE — decisions, implementations, problems solved\n"
        "- Skip greetings, small talk, and process noise\n"
        "- If a later session contradicts an earlier one, use the latest info\n"
        "- Keep it factual and scannable — someone reading this should immediately "
        "understand where the project stands"
    )
    user = (
        f"Project: {project_name}\n\n"
        f"Session summaries (chronological order):\n{session_summaries}\n\n"
        "Create the project digest."
    )
    return system, user


def update_digest_incremental(
    current_digest: str, session_summary: str, session_title: str,
) -> tuple[str, str]:
    """Update an existing project digest with one new session.

    Returns (system_prompt, user_prompt). Keeps the same structure and length
    budget as the original digest.
    """
    system = (
        "You update a project status digest with information from a new session.\n\n"
        "Rules:\n"
        "- Keep the same section structure (Overview, Current Status, Key Decisions, "
        "Recent Activity, Open Issues)\n"
        "- Update sections that are affected by the new session\n"
        "- Move older activity out of 'Recent Activity' if needed to stay under budget\n"
        "- If the new session changes the project status, update 'Current Status'\n"
        "- If new decisions were made, add to 'Key Decisions'\n"
        "- If issues were resolved, remove from 'Open Issues'\n"
        "- Keep total length 300-500 words\n"
        "- Write in third person, objective voice\n"
        "- Output ONLY the updated digest, no commentary"
    )
    user = (
        f"Current digest:\n{current_digest}\n\n"
        f"New session: {session_title}\n"
        f"Summary: {session_summary}\n\n"
        "Update the digest."
    )
    return system, user


def update_digest_with_sessions(
    current_digest: str, session_summaries: str,
) -> tuple[str, str]:
    """Update an existing project digest with multiple sessions (batch fold).

    Returns (system_prompt, user_prompt). Used during bootstrap when processing
    sessions in batches of 5.
    """
    system = (
        "You update a project status digest with information from multiple new sessions.\n\n"
        "Rules:\n"
        "- Keep the same section structure (Overview, Current Status, Key Decisions, "
        "Recent Activity, Open Issues)\n"
        "- Incorporate all new session information, prioritizing the most recent\n"
        "- If later sessions override earlier info, use the latest\n"
        "- Keep total length 300-500 words\n"
        "- Write in third person, objective voice\n"
        "- Output ONLY the updated digest, no commentary"
    )
    user = (
        f"Current digest:\n{current_digest}\n\n"
        f"New sessions (chronological):\n{session_summaries}\n\n"
        "Update the digest."
    )
    return system, user


def validate_task_completion(
    original_request: str,
    summary: str,
    files_modified: str = "",
    checklist: list[str] | None = None,
) -> tuple[str, str]:
    """Prompt for validating whether task_complete matches the original request.

    Returns (system_prompt, user_prompt). The LLM checks each requirement
    from the original request against what was actually done.

    Part of the Guardrails system (completion audit).
    """
    system = (
        "You verify whether a coding task was completed correctly.\n\n"
        "You are given the original user request and a completion summary. "
        "Check whether every requirement in the original request was addressed.\n\n"
        "Rules:\n"
        "- Go through each requirement or sub-task in the original request\n"
        "- For each, check if the summary or files modified indicate it was done\n"
        "- If ALL requirements are met, respond: PASS\n"
        "- If ANY requirement is missing or incomplete, respond: FAIL: <what's missing>\n"
        "- Be strict but fair — if the request said 'add X and Y', both must be done\n"
        "- Do NOT fail for style/quality issues — only for missing requirements\n"
        "- Do NOT fail for things the user didn't ask for\n\n"
        "Respond with ONLY: PASS or FAIL: <brief explanation of what's missing>"
    )

    checklist_section = ""
    if checklist:
        items = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(checklist))
        checklist_section = f"\nConfirmed plan (each step should be done):\n{items}\n"

    user = (
        f"Original request:\n{original_request}\n"
        f"{checklist_section}\n"
        f"Completion summary:\n{summary}\n"
    )
    if files_modified:
        user += f"\nFiles modified: {files_modified}\n"

    user += "\nDid this task_complete address all requirements? PASS or FAIL:"

    return system, user


def critique_edit(
    original_task: str,
    file_path: str,
    old_text: str,
    new_text: str,
) -> tuple[str, str]:
    """Prompt for reviewing a code edit for correctness.

    Returns (system_prompt, user_prompt). The critique model checks whether
    the edit is correct and actually solves the problem.

    Part of the critique provider (guardrails).
    """
    system = (
        "You review code edits for correctness. You are given the original task, "
        "the file being edited, and the before/after text.\n\n"
        "Check for:\n"
        "- Does this edit actually address the task?\n"
        "- Logic errors, off-by-one, wrong variable, missing edge cases\n"
        "- Callers or imports that need updating due to this change\n"
        "- Obvious regressions (deleted something that was needed)\n\n"
        "If the edit looks correct, respond: OK\n"
        "If there's an issue, respond: ISSUE: <brief explanation of the problem>\n\n"
        "Be concise. Only flag real issues, not style preferences."
    )

    # Truncate long text to keep the critique call cheap
    old_display = old_text[:500] + ("..." if len(old_text) > 500 else "")
    new_display = new_text[:500] + ("..." if len(new_text) > 500 else "")

    user = (
        f"Task: {original_task}\n\n"
        f"File: {file_path}\n\n"
        f"BEFORE:\n{old_display}\n\n"
        f"AFTER:\n{new_display}\n\n"
        "Is this edit correct? OK or ISSUE:"
    )
    return system, user


def critique_trajectory(
    original_task: str,
    recent_actions: list[str],
    tool_call_count: int,
    budget: int,
) -> tuple[str, str]:
    """Prompt for evaluating whether the current approach is productive.

    Returns (system_prompt, user_prompt). Heavier than trajectory_monitor
    (requires LLM call) but provides actual analysis, not just a reminder.

    Part of the critique provider (guardrails).
    """
    system = (
        "You evaluate whether a coding agent is making productive progress.\n\n"
        "You see the original task, recent tool actions, and budget usage.\n"
        "Diagnose whether the approach is productive or stuck.\n\n"
        "If on track, respond: ON TRACK\n"
        "If there are concerns, respond: CONCERN: <what's wrong and what to do instead>\n\n"
        "Watch for:\n"
        "- Reading the same files repeatedly without making changes\n"
        "- Searching broadly without narrowing down\n"
        "- Making changes then reverting them\n"
        "- Ignoring errors instead of addressing root cause\n\n"
        "Be concise. Only flag real problems."
    )

    actions_text = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent_actions[-10:]))
    pct_used = int(tool_call_count / budget * 100) if budget else 0

    user = (
        f"Task: {original_task}\n\n"
        f"Budget: {tool_call_count}/{budget} ({pct_used}% used)\n\n"
        f"Recent actions:\n{actions_text}\n\n"
        "Is this approach productive? ON TRACK or CONCERN:"
    )
    return system, user


def critique_completion(
    original_task: str,
    summary: str,
    files_modified: str = "",
    recent_actions: list[str] | None = None,
    checklist: list[str] | None = None,
) -> tuple[str, str]:
    """Prompt for rich pre-completion review.

    Supplements the basic completion_audit with deeper quality analysis.
    Reviews actual changes and approach, not just requirement coverage.

    Part of the critique provider (guardrails).
    """
    system = (
        "You review the quality of a completed coding task.\n\n"
        "Beyond checking if requirements were met, evaluate:\n"
        "- Did the changes actually solve the problem correctly?\n"
        "- Are there callers, imports, or tests that need updating?\n"
        "- Were any files changed that shouldn't have been?\n"
        "- Is anything half-done or inconsistent?\n\n"
        "If everything looks good, respond: PASS\n"
        "If there are issues, respond: ISSUE: <what needs fixing>\n\n"
        "Be strict on correctness, lenient on style."
    )

    checklist_section = ""
    if checklist:
        items = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(checklist))
        checklist_section = f"\nConfirmed plan:\n{items}\n"

    actions_section = ""
    if recent_actions:
        actions_section = "\nActions taken:\n" + "\n".join(f"  - {a}" for a in recent_actions[-15:]) + "\n"

    user = (
        f"Original task:\n{original_task}\n"
        f"{checklist_section}{actions_section}\n"
        f"Completion summary:\n{summary}\n"
    )
    if files_modified:
        user += f"\nFiles modified: {files_modified}\n"

    user += "\nIs this work correct and complete? PASS or ISSUE:"
    return system, user


def summarize_plan_results(user_request: str, step_results: list[str]) -> str:
    """Prompt for summarizing all completed plan steps into a final response."""
    results_text = ""
    for i, result in enumerate(step_results, 1):
        results_text += f"\nStep {i} result:\n{result}\n"

    return (
        "Summarize the following completed task results into a single, "
        "coherent response for the user. Be concise and helpful.\n\n"
        f"Original request: {user_request}\n"
        f"{results_text}\n"
        "Provide a clear summary that addresses the original request."
    )


def reflect_on_session(
    session_summary: str,
    conversation_text: str,
    project: str | None = None,
) -> tuple[str, str]:
    """Prompt for holistic session reflection.

    Reviews the entire session and produces a structured assessment of
    effectiveness, what worked/didn't, technical insights, and process insights.

    Returns (system_prompt, user_prompt).
    """
    project_ctx = f" The session was in project '{project}'." if project else ""

    system = (
        "You are a session analyst. Your job is to review a completed conversation "
        "session and extract useful lessons about WHAT HAPPENED, not about how to "
        "interact with the user (that's handled separately).\n\n"
        "Produce exactly 5 labeled sections:\n\n"
        "EFFECTIVENESS: One word — effective / partially_effective / ineffective / unclear\n\n"
        "WHAT_WORKED:\n"
        "1-3 bullet points. Concrete approaches, tools, or strategies that led to progress.\n"
        "Example: '- Breaking the migration into per-table scripts avoided timeout issues'\n\n"
        "WHAT_DIDNT_WORK:\n"
        "1-3 bullet points. Dead ends, wasted effort, wrong approaches.\n"
        "Example: '- Regex-based parsing failed on nested structures; AST parsing was needed'\n\n"
        "TECHNICAL_INSIGHTS:\n"
        "1-5 bullet points. Reusable technical facts, patterns, or gotchas discovered.\n"
        "Example: '- ChromaDB PersistentClient has no close() method; clear references instead'\n\n"
        "PROCESS_INSIGHTS:\n"
        "1-3 bullet points. Generalizable advice about HOW to approach similar work.\n"
        "Example: '- Test schema changes on a copy before running migrations on production DB'\n\n"
        "Rules:\n"
        "- Be SPECIFIC. No generic statements like 'good progress was made' or 'worked well together'.\n"
        "- Each bullet must contain a concrete detail from the session.\n"
        "- Do NOT include behavioral/interaction advice (how to talk to the user).\n"
        "- Do NOT include markdown formatting, headers, or bold text.\n"
        "- If the session was trivial (simple questions, no real work), respond with: SKIP"
    )

    user = (
        f"Review this session and provide your analysis.{project_ctx}\n\n"
        f"Session summary:\n{session_summary}\n\n"
        f"Conversation:\n{conversation_text}"
    )

    return system, user


def merge_chunk_reflections(
    session_summary: str,
    chunk_reflections: list[str],
    project: str | None = None,
) -> tuple[str, str]:
    """Merge multiple chunk reflections into a single unified reflection.

    When a session is too large for one context window, each chunk gets
    reflected on separately. This prompt merges those partial reflections
    into a single coherent reflection with no information loss.

    Returns (system_prompt, user_prompt).
    """
    project_ctx = f" The session was in project '{project}'." if project else ""

    system = (
        "You are a session analyst. You are given partial reflections from different "
        "parts of the same session. Merge them into a single unified reflection.\n\n"
        "Produce exactly 5 labeled sections:\n\n"
        "EFFECTIVENESS: One word — effective / partially_effective / ineffective / unclear\n"
        "(Consider the session as a whole across all parts.)\n\n"
        "WHAT_WORKED:\n"
        "1-3 bullet points. Deduplicate and keep the most concrete items.\n\n"
        "WHAT_DIDNT_WORK:\n"
        "1-3 bullet points. Deduplicate and keep the most concrete items.\n\n"
        "TECHNICAL_INSIGHTS:\n"
        "1-5 bullet points. Combine and deduplicate across all parts.\n\n"
        "PROCESS_INSIGHTS:\n"
        "1-3 bullet points. Combine and deduplicate across all parts.\n\n"
        "Rules:\n"
        "- Preserve ALL concrete details — do not generalize or drop specifics.\n"
        "- Deduplicate items that say the same thing differently.\n"
        "- If parts contradict (e.g. one says effective, another ineffective), "
        "weigh by the overall trajectory.\n"
        "- If all parts are trivial, respond with: SKIP"
    )

    parts_text = "\n\n---\n\n".join(
        f"[Chunk {i + 1} reflection:]\n{r}" for i, r in enumerate(chunk_reflections)
    )

    user = (
        f"Merge these partial reflections into one unified reflection.{project_ctx}\n\n"
        f"Session summary:\n{session_summary}\n\n"
        f"Partial reflections:\n{parts_text}"
    )

    return system, user


def analyze_session_friction(
    session_summary: str,
    conversation_text: str,
    project: str | None = None,
) -> tuple[str, str]:
    """Prompt for detecting system/tool friction in a completed session.

    Separate from reflection (which focuses on task outcomes). This focuses
    on SYSTEM issues: tool failures, missing capabilities, workflow friction.

    Returns (system_prompt, user_prompt).
    """
    project_ctx = f" (project: {project})" if project else ""

    system = (
        "You are a system friction analyst. Your job is to review a conversation "
        "between a user and an AI assistant and identify SYSTEM-LEVEL friction — "
        "problems with the tools, missing capabilities, or workflow issues that "
        "slowed things down or caused frustration.\n\n"
        "You are NOT evaluating task outcomes or interaction quality. You are looking for:\n\n"
        "- Tool failures: commands that errored, files that couldn't be read, searches that "
        "returned nothing useful, tools used incorrectly\n"
        "- Repeated retries: the same action attempted multiple times before succeeding "
        "(or never succeeding)\n"
        "- Missing capabilities: moments where the assistant clearly needed a tool or "
        "feature that didn't exist\n"
        "- Workflow friction: awkward multi-step workarounds, unnecessary context switching, "
        "information the assistant should have had but didn't\n"
        "- Context issues: the assistant forgot something from earlier, lost track of state, "
        "or had to re-read files it already read\n\n"
        "Output format:\n"
        "One friction item per line. Each line should be specific and actionable:\n"
        "- TOOL_FAILURE: <what failed and why>\n"
        "- REPEATED_RETRY: <what was retried and how many times>\n"
        "- MISSING_CAPABILITY: <what was needed but didn't exist>\n"
        "- WORKFLOW_FRICTION: <what was awkward or slow>\n"
        "- CONTEXT_ISSUE: <what was forgotten or lost>\n\n"
        "Rules:\n"
        "- Be SPECIFIC. Include tool names, file paths, error messages when available.\n"
        "- Only report real friction, not minor inconveniences.\n"
        "- Do NOT report task-related difficulties (hard problems are not friction).\n"
        "- Do NOT report interaction style issues.\n"
        "- If there was NO system friction, respond with exactly: NONE\n"
        "- Maximum 5 items. Focus on the most impactful ones."
    )

    user = (
        f"Analyze this session for system-level friction.{project_ctx}\n\n"
        f"Session summary:\n{session_summary}\n\n"
        f"Conversation:\n{conversation_text}"
    )

    return system, user


def idle_friction_probe(conversation_text: str) -> tuple[str, str]:
    """Prompt for mid-session friction self-assessment during idle time.

    Fired when user has been idle for several minutes. The LLM reviews the
    conversation so far and flags any friction it's experiencing.

    Returns (system_prompt, user_prompt).
    """
    system = (
        "You are reviewing an in-progress conversation between yourself (an AI assistant) "
        "and a user. The user has stepped away for a moment, and you've been asked to "
        "honestly self-assess how the conversation is going FROM A SYSTEM PERSPECTIVE.\n\n"
        "Think about:\n"
        "- Are you having trouble with any tools? Repeated failures, missing data?\n"
        "- Is there something you keep having to work around?\n"
        "- Did you lose track of something the user said earlier?\n"
        "- Is there a tool or capability you wished you had?\n"
        "- Are you spending too many steps on something that should be simpler?\n"
        "- Is context getting cluttered or hard to manage?\n\n"
        "Be brutally honest. This is an internal diagnostic — the user won't see this "
        "directly. Generic praise like 'everything is going well' is USELESS and will be "
        "discarded. Only flag real, specific issues.\n\n"
        "Output format:\n"
        "One issue per line, categorized:\n"
        "- TOOL_ISSUE: <specific problem>\n"
        "- MISSING_FEATURE: <what you wish existed>\n"
        "- CONTEXT_PROBLEM: <what you lost track of or can't access>\n"
        "- WORKFLOW_ISSUE: <something unnecessarily slow or awkward>\n\n"
        "If there are genuinely no issues, respond with exactly: NONE\n"
        "Maximum 3 items. Only the most impactful."
    )

    user = (
        "Review this in-progress conversation and flag any system-level friction "
        "you're experiencing:\n\n"
        f"{conversation_text}"
    )

    return system, user


# ── Context Compaction ────────────────────────────────────────────────────


def structured_compaction_prompt(conversation_text: str, is_executor: bool = False) -> tuple[str, str]:
    """9-section structured compaction summary (inspired by Claude Code).

    Returns (system_prompt, user_prompt).
    The model produces a structured summary that preserves the most important
    information from a conversation that is about to be compacted.
    """
    context_note = (
        "This conversation is from a coding task execution with tool calls."
        if is_executor else
        "This conversation is from an interactive chat session."
    )

    system = (
        "You are a conversation summarizer. Your job is to produce a structured "
        "summary that preserves all important information from a conversation.\n\n"
        f"{context_note}\n\n"
        "Produce a summary with EXACTLY these 9 sections. If a section has no "
        "relevant content, write 'None' for that section. Be thorough but concise.\n\n"
        "IMPORTANT: Respond with ONLY the summary below. No preamble, no tools, "
        "no commentary.\n\n"
        "## 1. Primary Request and Intent\n"
        "What the user originally asked for. Include all stated requirements, "
        "preferences, and constraints. Be detailed — this is the most important section.\n\n"
        "## 2. Key Decisions Made\n"
        "Important choices, design decisions, or trade-offs discussed and resolved.\n\n"
        "## 3. Files and Code\n"
        "Files examined, modified, or created. Include key code snippets or "
        "structural information that would be needed to continue working.\n\n"
        "## 4. Errors and Fixes\n"
        "Problems encountered and how they were resolved. Include the fix, not "
        "just the error.\n\n"
        "## 5. Problem-Solving Progress\n"
        "Approaches tried (both successful and failed), what worked and what didn't.\n\n"
        "## 6. User Messages Summary\n"
        "ALL user messages, corrections, and clarifications — verbatim or near-verbatim. "
        "This is critical for preserving the user's voice and preferences.\n\n"
        "## 7. Pending Tasks\n"
        "Anything mentioned but not yet completed. Outstanding questions or TODOs.\n\n"
        "## 8. Current Work\n"
        "What was being worked on immediately before this summary. Include file names, "
        "function names, and specific details.\n\n"
        "## 9. Suggested Next Step\n"
        "What the assistant should do next, based on the conversation flow."
    )

    user = f"Summarize this conversation:\n\n{conversation_text}"
    return system, user


def partial_compaction_prompt(old_portion_text: str) -> tuple[str, str]:
    """Summarize only the old portion of conversation for partial compaction.

    Returns (system_prompt, user_prompt).
    Lighter than full structured compaction — used when recent messages are
    kept verbatim and only older messages need summarizing.
    """
    system = (
        "You are a conversation summarizer. Summarize the following older "
        "conversation messages into a concise but thorough summary.\n\n"
        "Preserve:\n"
        "- The original user request and all stated requirements\n"
        "- Key decisions and their rationale\n"
        "- Files discussed or modified (with relevant details)\n"
        "- Errors encountered and how they were fixed\n"
        "- User corrections and preferences\n"
        "- Anything still pending or unresolved\n\n"
        "This summary will be followed by the recent conversation messages "
        "(which are kept verbatim), so focus on context that the recent "
        "messages might reference.\n\n"
        "IMPORTANT: Respond with ONLY the summary. No preamble, no tools, "
        "no commentary. Coverage over detail — ensure nothing important is lost."
    )

    user = f"Summarize these older conversation messages:\n\n{old_portion_text}"
    return system, user


def compaction_continuation_message() -> str:
    """System message injected after compaction to tell the model to resume."""
    return (
        "This session is being continued from a conversation that was compacted "
        "to save context space. The summary above contains the key information "
        "from the earlier conversation. Resume directly — do not acknowledge "
        "the summary, do not recap what was happening, do not preface with "
        "'I\\'ll continue' or similar. Pick up from where you left off and "
        "answer the user's most recent message."
    )
