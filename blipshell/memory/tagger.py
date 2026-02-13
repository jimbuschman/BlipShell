"""Tag extraction (direct port of Tagging.cs).

Extracts topic tags, behavior tags, and background triggers from messages.
"""

import re

# Topic patterns: tag_name -> list of regex patterns
TOPIC_PATTERNS: dict[str, list[str]] = {
    "c#": [r"\bc#\b", r"\.NET", r"asp\.net", r"razor"],
    "sql": [r"\bsql\b", r"\bquery\b", r"sqlite", r"dapper", r"SELECT\s+\*?\s*FROM"],
    "python": [r"\bpython\b", r"\bpip\b", r"\bwhisper\b", r"huggingface"],
    "javascript": [r"\bjavascript\b", r"\bjs\b", r"node\.js"],
    "html": [r"\bhtml\b", r"html page", r"markup"],
    "razor-pages": [r"razor[- ]pages", r"asp\.net razor"],
    "signalr": [r"\bsignalr\b"],
    "dapper": [r"\bdapper\b"],
    "sqlite": [r"\bsqlite\b"],
    "ollama": [r"\bollama\b"],
    "ngrok": [r"\bngrok\b"],
    "cloudflare-tunnel": [r"cloudflare.*tunnel"],
    "llm": [r"\bllm\b", r"language model", r"mistral", r"llama"],
    "embedding": [r"\bembedding\b", r"\bembed\b", r"\bvector\b"],
    "vector-search": [r"vector search", r"similarity search"],
    "tagging": [r"\btagging\b", r"message tagging"],
    "summarization": [r"summarize", r"summary"],
    "chat-completion": [r"chat completion", r"openai chat", r"streaming reply"],
    "prompt-design": [r"\bprompt\b", r"\binstruction\b", r"\bprompting\b"],
    "context-injection": [r"context injection", r"inject.*memory"],
    "memory-retrieval": [r"retrieve.*memory", r"search.*memory"],
    "fine-tuning": [r"fine[- ]tuning", r"finetune"],
    "identity-framework": [r"identity framework", r"echo framework", r"trait design"],
    "system-architecture": [r"architecture", r"system design", r"modular structure"],
    "injection-logic": [r"context injection", r"inject-when", r"trigger rules"],
    "autonomy": [r"autonomy", r"self[- ]direction", r"self[- ]guided"],
    "identity": [r"\bidentity\b", r"who am i", r"self[- ]definition"],
    "memory-system": [r"memory system", r"core memory", r"lesson system"],
    "lesson-system": [r"\blesson\b", r"learning loop", r"lesson system"],
    "reflection-loop": [r"reflection loop", r"feedback loop", r"self[- ]reflection"],
    "meta-learning": [r"meta[- ]learning"],
    "self-awareness": [r"self[- ]awareness", r"selfhood"],
    "drift-detection": [r"\bdrift\b", r"off[- ]track", r"misaligned"],
    "task-queue": [r"task queue", r"message queue", r"job queue"],
    "session-history": [r"session history", r"past conversation"],
    "log-analysis": [r"\blog\b", r"log analysis", r"logging"],
    "ui-design": [r"\bui\b", r"\binterface\b", r"chat window"],
    "console-app": [r"console app", r"cmd line", r"terminal"],
    "chroma": [r"\bchroma\b", r"chromadb"],
    "query-optimization": [r"query.*optimiz", r"sql tuning"],
    "background-processing": [r"background thread", r"\basync\b", r"idle process"],
    "threading": [r"\bthread\b", r"\bconcurrency\b"],
    "bug": [r"\bbug\b", r"\bissue\b", r"something's wrong"],
    "debug": [r"\bdebug\b", r"\btrace\b", r"step through"],
    "test-run": [r"test run", r"unit test", r"\btesting\b"],
    "performance": [r"\bperformance\b", r"\blag\b", r"\bslow\b", r"\blatency\b"],
    "timeout": [r"\btimeout\b", r"\bhang\b", r"\bfreeze\b"],
    "note": [r"note to self", r"i should remember", r"meta note"],
    "journal": [r"\bjournal\b", r"reflective entry"],
    "design-doc": [r"design doc", r"architecture plan"],
    "annotation": [r"\bannotate\b", r"\bcomment\b"],
    "documentation": [r"\bdocumentation\b", r"\bdocs\b", r"\breadme\b"],
    "code": [r"```", r"public static", r"using System", r"\bdef ", r"function\b",
             r"\bclass\b", r"namespace\b", r"#include"],
}

# Behavior tag patterns: tag_name -> (strong_triggers, soft_triggers)
# Strong triggers get score +2, soft triggers get score +1
BEHAVIOR_PATTERNS: dict[str, tuple[list[str], list[str]]] = {
    "identity": (
        ["sense of self", "who I am", "selfhood", "core identity"],
        ["identity", "i am", "self"],
    ),
    "emotion": (
        ["emotional awareness", "frustrated", "i feel", "emotions"],
        ["feelings", "frustration", "emotional"],
    ),
    "core-memory": (
        ["core memory", "permanent memory"],
        ["important memory", "persistent memory"],
    ),
    "reflection": (
        ["self-reflection", "learning loop", "lesson system"],
        ["feedback loop", "reflection", "learning from mistakes"],
    ),
    "goal": (
        ["guiding star", "primary objective"],
        ["goal", "priority", "focus"],
    ),
    "task": (
        ["assigned task", "task queue"],
        ["task", "to-do", "next step"],
    ),
    "drift": (
        ["drifted off", "autopilot behavior", "lost focus"],
        ["drift", "off-track", "misaligned"],
    ),
    "memory-system": (
        ["memory system", "injected memory", "memory recall"],
        ["context memory", "session memory", "retention system"],
    ),
    "autonomy": (
        ["autonomy", "self-guided", "independent system"],
        ["self-driven", "self-direction", "freedom to learn"],
    ),
    "identity-framework": (
        ["identity framework", "trait model", "echo framework"],
        ["core traits", "alignment loop", "constitution"],
    ),
    "architecture": (
        ["system architecture", "overall structure", "modular design"],
        ["framework", "system design", "structure"],
    ),
    "injection-logic": (
        ["memory injection", "context injection", "inject when"],
        ["context logic", "trigger rules"],
    ),
}

# Background trigger patterns for detecting actionable items
BACKGROUND_TRIGGER_PATTERNS: dict[str, list[str]] = {
    "research-idea": [
        "we should look into", "worth researching", "curious if",
        "this could be researched", "let's test", "we might try",
        "investigate", "experiment with", "dig deeper",
    ],
    "unfinished-thought": [
        "i'm not sure", "i need to think more", "come back to this",
        "not fully formed", "needs more thought", "unclear right now",
        "needs exploration", "this isn't final", "to be continued",
        "we'll revisit this", "something to think about",
    ],
    "reflection-point": [
        "something to reflect on", "lesson here", "learning moment",
        "note to self", "worth considering", "insight", "pause and think",
        "might be meaningful", "need to internalize",
    ],
    "task-candidate": [
        "we should", "need to", "to-do", "eventually",
        "add to list", "remember to", "put this on the queue",
        "pending task", "assign this later",
    ],
    "lesson-candidate": [
        "i've learned", "lesson here", "what i realized", "going forward i should",
        "this taught me", "next time i'll", "i need to remember",
        "important takeaway", "i understand now", "this shows me that",
    ],
    "loop-closure": [
        "i see what you mean", "that makes sense now", "now i understand",
        "thanks for clarifying", "so the answer is", "i've updated my thinking",
        "that closes the loop", "we've resolved that", "this wraps it up",
    ],
}


def tag_topics(message: str) -> list[str]:
    """Extract topic tags from a message using regex patterns."""
    tags = set()
    for tag_name, patterns in TOPIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, message, re.IGNORECASE):
                tags.add(tag_name)
                break
    return list(tags)


def tag_behavior(message: str, confidence_threshold: int = 1) -> list[str]:
    """Extract behavior tags from a message using strong/soft triggers.

    Strong triggers add +2 to score, soft triggers add +1.
    Tags with score >= confidence_threshold are included.
    """
    scores: dict[str, int] = {}

    for tag_name, (strong, soft) in BEHAVIOR_PATTERNS.items():
        for trigger in strong:
            if trigger.lower() in message.lower():
                scores[tag_name] = scores.get(tag_name, 0) + 2

        for trigger in soft:
            pattern = rf"\b{re.escape(trigger)}\b"
            if re.search(pattern, message, re.IGNORECASE):
                scores[tag_name] = scores.get(tag_name, 0) + 1

    result = [tag for tag, score in scores.items() if score >= confidence_threshold]
    return result if result else ["neutral"]


def detect_background_triggers(message: str) -> list[str]:
    """Detect background actionable triggers in a message."""
    triggers = set()
    for tag_name, phrases in BACKGROUND_TRIGGER_PATTERNS.items():
        for phrase in phrases:
            if phrase.lower() in message.lower():
                triggers.add(tag_name)
                break
    return list(triggers)


def tag_message(message: str, max_tags: int = 7) -> list[str]:
    """Combined tagger: behavior tags + topic tags, capped at max_tags.

    Port of CombinedTagger.TagMessage().
    Behavior tags are prioritized over topic tags.
    """
    topic_tags = tag_topics(message)
    behavior_tags = tag_behavior(message)

    # Behavior tags first, then topics, deduplicated
    all_tags = list(dict.fromkeys(behavior_tags + topic_tags))

    # Remove "neutral" if other tags are present
    if len(all_tags) > 1 and "neutral" in all_tags:
        all_tags.remove("neutral")

    # Cap total tags, prioritizing behavior over topic
    if len(all_tags) > max_tags:
        behavior_set = set(behavior_tags)
        topic_only = [t for t in topic_tags if t not in behavior_set]
        all_tags = (behavior_tags + topic_only)[:max_tags]

    return all_tags
