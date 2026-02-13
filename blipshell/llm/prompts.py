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
        "Summarize the following memory in 1 concise, factual sentence. "
        "Avoid lists or multiple versions. Focus on core details.\n\n"
        f"Memory: {text}"
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
    """Prompt for extracting lessons from a conversation."""
    return (
        "Looking at the following conversation:\n"
        "1. Did the assistant understand the user's intention?\n"
        "2. What did it miss?\n"
        "3. What 3-5 lessons should it internalize to improve in the future?\n"
        "4. Growth Trajectory?\n\n"
        "Please separate your response into:\n"
        "Evaluation Summary\n"
        "Lessons\n\n"
        f"Conversation: {text}"
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
