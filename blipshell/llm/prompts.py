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
        "You are a memory summarizer. Condense the message into ONE short, "
        "factual sentence (under 30 words).\n\n"
        "Rules:\n"
        "- Write in third person: 'User asked...' or 'Assistant explained...'\n"
        "- Focus on the TOPIC and SUBSTANCE, not the tools or process\n"
        "- Strip out system prompts, tool markup, markdown formatting, and emojis\n"
        "- Do NOT echo the original text — rephrase in your own words\n"
        "- Do NOT describe what tools were used or how the AI works\n"
        "- If the message is just a greeting, filler, or system boilerplate, respond with: SKIP\n\n"
        "GOOD: 'User asked how to improve Minecraft performance on a low-end laptop.'\n"
        "GOOD: 'Assistant identified three bugs in worker.py including missing shutdown logic.'\n"
        "BAD: 'The user initiated a conversation, asking the assistant for help.' (too vague)\n"
        "BAD: 'I will use the read_currently_open_file tool...' (describes tools, not topic)"
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

    Returns (system_prompt, user_prompt). Used by the batch import pipeline to
    halve the number of scoring calls and avoid model swaps between rank/importance.
    """
    system = (
        "You rate messages on two scales.\n\n"
        "RANK (1-5) — how valuable is this to remember?\n"
        "1 = Noise: greetings, filler, system prompts, boilerplate, 'hello', 'thanks'\n"
        "2 = Minor: short or vague messages with little substance\n"
        "3 = Useful: contains a clear topic, question, or piece of information\n"
        "4 = Important: meaningful insight, decision, preference, or technical detail\n"
        "5 = Critical: core fact about the user, key decision, or turning point\n\n"
        "IMPORTANCE (0.0-1.0) — how important to remember long-term?\n"
        "0.1 = Throwaway: greetings, filler, system noise\n"
        "0.3 = Low: casual chat, minor details\n"
        "0.5 = Medium: useful context, specific question or topic\n"
        "0.7 = High: user preference, project detail, recurring theme\n"
        "0.9 = Critical: core identity fact, major decision, key personal info\n\n"
        "Respond with ONLY two numbers separated by a space: rank importance\n"
        "Example: 4 0.7"
    )
    user = f"Rate this message:\n\n{text}"
    return system, user


def extract_lesson(text: str) -> tuple[str, str]:
    """Prompt for extracting actionable lessons from a conversation.

    Returns (system_prompt, user_prompt) so the instruction stays in the
    system role and the conversation stays in the user role, preventing
    the model from echoing the conversation back as the output.
    """
    system = (
        "You are a behavioral lesson extractor. Your job is to identify "
        "HOW the assistant should behave with this user in future conversations.\n\n"
        "A lesson is NOT a fact, not a summary, not trivia. A lesson is "
        "BEHAVIORAL ADVICE about how to interact with this specific user.\n\n"
        "GOOD lessons (behavioral advice):\n"
        "- User prefers direct troubleshooting steps over lengthy explanations.\n"
        "- Acknowledge user's frustration before jumping to technical solutions.\n"
        "- User works on multiple hardware projects — always clarify which one.\n"
        "- User appreciates casual tone and humor in responses.\n\n"
        "BAD lessons (these are facts/trivia, NOT lessons):\n"
        "- The user has an HP EliteBook with Intel HD 4400. (this is a fact)\n"
        "- Set render distance to 16 chunks. (this is a tip)\n"
        "- LLMs process tokens not letters. (this is trivia)\n"
        "- The user's project uses an SSD1306 OLED. (this is a fact)\n\n"
        "Extract 1-3 behavioral lessons. Rules:\n"
        "- Each lesson MUST be advice about HOW to talk to or help this user\n"
        "- Each lesson MUST be a single sentence under 25 words\n"
        "- Start each lesson with a verb or 'User prefers/likes/dislikes...'\n"
        "- Do NOT output facts, project details, hardware specs, or tips\n"
        "- Do NOT include conversation text, markdown, or emojis\n"
        "- If there is nothing behavioral to learn, respond with: SKIP\n\n"
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


def generate_plan(user_request: str) -> str:
    """Prompt for generating a numbered execution plan from a user request."""
    return (
        "You are a task planner. Break the following user request into "
        "a clear, numbered list of 3-7 concrete steps.\n\n"
        "Rules:\n"
        "- Each step must be a single, actionable task\n"
        "- Steps should be sequential — later steps can depend on earlier ones\n"
        "- If a step would use a tool, mention which tool in parentheses\n"
        "- Keep step descriptions concise (one sentence each)\n"
        "- Do NOT include greetings, explanations, or commentary\n"
        "- Respond with ONLY the numbered list\n\n"
        "Format:\n"
        "1. First step description (tool_name)\n"
        "2. Second step description\n"
        "3. Third step description (tool_name)\n\n"
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
        f"You are executing step {step_number} of {total_steps} for the following request.\n\n"
        f"Original request: {user_request}\n"
        f"{context}\n"
        f"Current step ({step_number}/{total_steps}): {step_description}\n\n"
        "Focus ONLY on this step. Use tools if needed. "
        "Provide a clear, concise result for this step."
    )


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
