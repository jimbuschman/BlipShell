"""Lightweight intent detection for user messages.

Pure-function classifiers used to decide when to apply special handling to a
turn — e.g. whether a message is a code-review/critique request that should be
grounded in an actual read/grep before the model states findings.

Modeled on the research-mode classifier in the CLI (`_detect_research_intent`):
conservative, regex-based, with strong signals (one is enough), weak signals
(need length + count), and action suppressors that veto detection. No LLM cost.
"""

import re

# Guidance injected into the system prompt (both chat paths) when a review
# request is detected. Single source of truth so the wording stays consistent.
REVIEW_GROUNDING_GUIDANCE = (
    "\n\n[REVIEW REQUEST]\n"
    "This asks you to evaluate code. Do NOT state findings (bugs, stale code, "
    "issues, things to improve) you have not grounded by actually reading or "
    "searching the relevant code this turn. For each finding, cite the file and "
    "line. If you haven't looked yet, look first. If you looked and found "
    "nothing, say so plainly — \"no issues found\" is a valid, correct answer."
)

# Strong review signals — any ONE match flags a review request. Phrase-based so
# they don't fire on bare action verbs (e.g. "refactor X" is a do-it task, but
# "refactor opportunities" is a review).
_REVIEW_STRONG = [
    re.compile(r"\breview\b", re.I),
    re.compile(r"\bcritique\b", re.I),
    re.compile(r"could be (improved|better)", re.I),
    re.compile(r"(what|anything|something|things?)\b.{0,30}\b(improve|improved|better)\b", re.I),
    re.compile(r"would you (improve|change|do differently)", re.I),
    re.compile(r"(suggestions?|ideas?|recommendations?)\b.{0,20}\b(improv|better|cleaner)", re.I),
    re.compile(r"(what'?s|whats) wrong with", re.I),
    re.compile(r"\bany (issues|bugs|problems|smells)\b", re.I),
    re.compile(r"feedback on", re.I),
    re.compile(r"code smell", re.I),
    re.compile(r"anti-?pattern", re.I),
    re.compile(r"refactor (opportunit|candidate)", re.I),
    re.compile(r"tech(nical)? debt", re.I),
    re.compile(r"audit (the |this |my |our )?(code|codebase|repo|file|project)", re.I),
    re.compile(r"look for (bugs|issues|problems)", re.I),
]

# Weak review signals — need a longer message AND (2+ hits OR 1 hit + "?").
_REVIEW_WEAK = [
    re.compile(r"is this (good|correct|right|ok|fine|clean)", re.I),
    re.compile(r"thoughts on", re.I),
    re.compile(r"how (does|is)\b.{0,30}\blook", re.I),
]

# Action/conversational suppressors — any ONE vetoes detection. These are do-it
# requests (the user wants work performed, not a critique) and status queries.
# Action verbs are anchored to the start of the message (after optional
# politeness) so that nouns like "the build system" or "the write path" don't
# falsely veto a genuine review. Deliberately omits "refactor" (handled as a
# phrase in _REVIEW_STRONG) and "check"/"look" (those can be review verbs).
_REVIEW_SUPPRESS = [
    re.compile(
        r"^(can you |could you |would you |please |pls |i want you to |i'?d like you to )*"
        r"(fix|add|create|build|implement|write|change|modify|remove|delete|rename|migrate|install|deploy|update)\b",
        re.I,
    ),
    re.compile(r"\b(show me|status|right now|currently)\b", re.I),
    re.compile(r"^(run|list)\b", re.I),
]


def detect_review_intent(message: str) -> bool:
    """Return True if the message reads as a code-review / critique request.

    Conservative by design: suppressors veto first (a do-it task is not a
    review), then one strong signal is enough, otherwise weak signals need a
    longer message plus multiple hits (or one hit and a question mark).
    """
    if not message or len(message.strip()) < 8:
        return False

    for p in _REVIEW_SUPPRESS:
        if p.search(message):
            return False

    for p in _REVIEW_STRONG:
        if p.search(message):
            return True

    if len(message) >= 50:
        weak_hits = sum(1 for p in _REVIEW_WEAK if p.search(message))
        if weak_hits >= 2:
            return True
        if weak_hits == 1 and message.strip().endswith("?"):
            return True

    return False
