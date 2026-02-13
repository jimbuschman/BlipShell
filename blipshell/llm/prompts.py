"""All LLM prompt templates centralized (port of LLMUtilityCalls.cs)."""

UTILITY_SYSTEM_PROMPT = (
    "You are a highly efficient, single-output processing module. "
    "Your ONLY purpose is to produce the requested output. "
    "You will NEVER engage in conversation, offer greetings, ask questions, "
    "or add any introductory or concluding remarks. "
    "Respond with nothing but the requested output."
)


def rank_memory(text: str) -> str:
    """Prompt for ranking a memory 1-5."""
    return (
        "You are evaluating a message to determine how informative or meaningful it is.\n\n"
        "Based on the content, assign it a rank from 1 to 5:\n\n"
        "1 - Noise / Fluff: Boilerplate, repetitive, off-topic, or lacking meaningful content.\n"
        "2 - Minor: Light emotional context or vague thought, lacks depth or specificity.\n"
        "3 - Useful: Contains at least one clear idea, insight, or point worth keeping.\n"
        "4 - Important: Clear relevance, meaningful insight, decision, realization, or reflective moment.\n"
        "5 - Critical: Core to identity, evolution, or decision-making. Key turning points.\n\n"
        "Respond with ONLY the rank (1-5).\n\n"
        f"Message: {text}"
    )


def rephrase_as_memory_style(text: str) -> str:
    """Prompt to rephrase a query as a declarative memory-style sentence."""
    return (
        "Rephrase the question as a direct, factual sentence someone might have said "
        "in a conversation. Avoid emotional or poetic language. Be concise and declarative.\n\n"
        f"Question: {text}\n"
        "Declarative:"
    )


def summarize_memory(text: str) -> str:
    """Prompt for summarizing a memory."""
    return (
        "Summarize the following conversation excerpt in 1 concise, factual sentence. "
        "Focus on what the USER said, asked, decided, or learned. "
        "Do NOT describe the assistant's capabilities, tools, or system features. "
        "Do NOT summarize meta-discussion about the AI itself. "
        "If the content is only about the AI system and not the user, respond with: SKIP\n\n"
        f"Excerpt: {text}"
    )


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


def ask_importance(text: str) -> str:
    """Prompt for rating memory importance 0.0-1.0."""
    return (
        "Rate the importance of the following memory on a scale from 0.0 to 1.0.\n"
        "Use the following guidelines:\n"
        "- 1.0 = Deeply personal, emotionally significant, critical fact, or core belief\n"
        "- 0.7 = Important context or recurring theme\n"
        "- 0.4 = Useful but minor detail\n"
        "- 0.1 = Casual, generic, or low-impact\n\n"
        "Respond ONLY with a single numeric value.\n\n"
        f"Memory: {text}"
    )


def extract_lesson(text: str) -> str:
    """Prompt for extracting actionable lessons from a conversation."""
    return (
        "Extract 1-3 short, actionable lessons from this conversation. "
        "Focus on:\n"
        "- User preferences (communication style, level of detail, topics of interest)\n"
        "- Facts about the user (name, projects, tools they use, technical level)\n"
        "- What worked well or poorly in the interaction\n"
        "- Technical insights or decisions made\n\n"
        "Rules:\n"
        "- Each lesson must be a single concise sentence\n"
        "- Write from the perspective of advice for the assistant\n"
        "- Do NOT write an evaluation, summary, or self-review\n"
        "- Do NOT include generic observations about AI capabilities\n"
        "- Only include lessons that are specific and actionable\n\n"
        "Format: One lesson per line, no numbering, no headers.\n\n"
        f"Conversation:\n{text}"
    )


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
