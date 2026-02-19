"""Query classification and dynamic pool budget profiles.

Classifies user messages into query profiles using regex heuristics (no LLM),
then computes per-pool token budgets so gather_memory() can allocate context
differently for recall-heavy, session-heavy, or coding queries.
"""

import re

# Pool percentage presets per query type
PROFILES: dict[str, dict[str, float]] = {
    "balanced": {
        "Core": 0.10,
        "ActiveSession": 0.35,
        "RecentHistory": 0.15,
        "Recall": 0.30,
        "Buffer": 0.10,
    },
    "recall": {
        "Core": 0.10,
        "ActiveSession": 0.20,
        "RecentHistory": 0.15,
        "Recall": 0.45,
        "Buffer": 0.10,
    },
    "session": {
        "Core": 0.08,
        "ActiveSession": 0.45,
        "RecentHistory": 0.17,
        "Recall": 0.20,
        "Buffer": 0.10,
    },
    "coding": {
        "Core": 0.15,
        "ActiveSession": 0.30,
        "RecentHistory": 0.10,
        "Recall": 0.35,
        "Buffer": 0.10,
    },
}

# --- Classification patterns ---

# Recall-heavy: questions about past info
_RECALL_PATTERNS = re.compile(
    r"(?:"
    r"^(?:what|when|where|who|which)\s+(?:was|were|is|did|do)\b"
    r"|^do you (?:remember|recall|know)\b"
    r"|^did (?:i|we) (?:ever|mention|say|tell|discuss)\b"
    r"|^(?:remind me|what did (?:i|we))\b"
    r"|\b(?:last time|previously|before|earlier|in the past)\b"
    r"|\b(?:my (?:name|cat|dog|pet|favorite|preference))\b"
    r")",
    re.IGNORECASE,
)

# Session-heavy: short follow-ups referencing current conversation
_SESSION_PATTERNS = re.compile(
    r"(?:"
    r"^(?:explain|elaborate|clarify|expand)\s+(?:that|this|more|further)\b"
    r"|^(?:what do you mean|can you clarify|go on|continue)\b"
    r"|^(?:yes|no|ok|sure|right|exactly|correct)\b"
    r"|\b(?:you (?:just |)(?:said|mentioned|suggested|recommended))\b"
    r"|\b(?:above|earlier in this conversation)\b"
    r")",
    re.IGNORECASE,
)

# Coding: programming-related queries
_CODING_PATTERNS = re.compile(
    r"(?:"
    r"\b(?:implement|debug|refactor|write (?:a |the |)(?:function|class|script|code|test|method))\b"
    r"|\b(?:fix (?:the |this |a |)(?:bug|error|issue|code))\b"
    r"|\b(?:python|javascript|typescript|rust|java|c\+\+|golang|sql)\b"
    r"|```"
    r"|\b(?:def |class |function |import |const |let |var )\b"
    r"|\b(?:api|endpoint|database|query|schema|migration)\b"
    r"|\b(?:compile|runtime|syntax|exception|traceback|stack trace)\b"
    r")",
    re.IGNORECASE,
)


def classify_query(message: str) -> str:
    """Classify a user message into a query profile. No LLM call.

    Returns one of: "recall", "session", "coding", "balanced".
    """
    stripped = message.strip()

    # Short messages (<15 chars) that aren't code are likely follow-ups
    if len(stripped) < 15 and "```" not in stripped:
        return "session"

    if _CODING_PATTERNS.search(stripped):
        return "coding"

    if _RECALL_PATTERNS.search(stripped):
        return "recall"

    if _SESSION_PATTERNS.search(stripped):
        return "session"

    return "balanced"


def compute_pool_budgets(
    profile_name: str,
    total_budget: int,
    hard_caps: dict[str, int | None],
) -> dict[str, int]:
    """Convert a profile's percentages into absolute token budgets.

    Respects hard caps — if a pool's percentage allocation exceeds its hard cap,
    the excess is redistributed proportionally to uncapped pools.
    """
    profile = PROFILES.get(profile_name, PROFILES["balanced"])

    budgets: dict[str, int] = {}
    excess = 0

    # First pass: apply percentages, cap where needed
    for pool_name, pct in profile.items():
        raw_budget = int(total_budget * pct)
        cap = hard_caps.get(pool_name)
        if cap and raw_budget > cap:
            excess += raw_budget - cap
            budgets[pool_name] = cap
        else:
            budgets[pool_name] = raw_budget

    # Second pass: redistribute excess to uncapped pools proportionally
    if excess > 0:
        uncapped = {
            name: budgets[name]
            for name in budgets
            if not hard_caps.get(name) or budgets[name] < hard_caps[name]
        }
        total_uncapped = sum(uncapped.values())
        if total_uncapped > 0:
            for name, budget in uncapped.items():
                share = int(excess * (budget / total_uncapped))
                cap = hard_caps.get(name)
                if cap:
                    budgets[name] = min(budget + share, cap)
                else:
                    budgets[name] = budget + share

    return budgets
