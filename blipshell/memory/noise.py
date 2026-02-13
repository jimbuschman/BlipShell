"""Noise filtering (direct port of NoiseCheck.cs).

Filters out low-value messages like greetings, reactions, and filler
before they enter the memory pipeline.
"""

import re

# Signal words that indicate potential meaningful content
SIGNAL_WORDS = {
    "you", "feel", "doing", "okay", "sure", "think", "remember",
    "what", "why", "still", "about", "mean", "want", "been",
}

# Known noise phrases (exact match after normalization)
NOISE_PHRASES = {
    # Greetings
    "hi", "hello", "hey", "hi there", "hey there", "yo",
    "whats up", "sup", "howdy", "good morning", "good afternoon", "good night",
    "bye", "goodbye", "see ya", "later", "take care", "gn", "night",
    # Short affirmatives/negatives
    "ok", "okay", "yeah", "nah", "maybe", "got it", "roger", "sure", "yup", "nope",
    "yes", "no", "alright", "right", "uh huh", "mm hmm", "mhm", "aye", "bet", "fine",
    "k", "kk",
    # Generic reactions
    "wow", "oh", "ah", "huh", "oops", "whoops", "hm", "hmm", "heh", "hmm ok",
    "okay then", "cool", "nice", "great", "awesome", "interesting", "noted",
    "makes sense", "understood",
    # Internet/text slang
    "lol", "haha", "lmao", "lmfao", "rofl", "smh", "brb", "btw", "idk", "imo",
    "imho", "tbh", "omg", "omfg", "ikr", "yeet", "fr", "nvm", "ffs", "wtf", "wth",
}

# Filler words (noise only when used alone)
FILLER_WORDS = {"um", "uh", "well", "like", "you know", "i mean"}

# Laughter pattern
LAUGHTER_PATTERN = re.compile(r"^(ha|lol|lmao|rofl)+[!]*$", re.IGNORECASE)

# Declarative sentence pattern
DECLARATIVE_START = re.compile(r"^(I|My|This|The|It|There is|\b[A-Z][a-z]+)\b")


def _normalize(text: str) -> str:
    """Normalize text for noise comparison."""
    lower = text.lower().strip()
    lower = re.sub(r"[^\w\s]", "", lower)  # remove punctuation
    lower = re.sub(r"\s+", " ", lower)  # normalize whitespace
    return lower


def contains_signal_words(text: str) -> bool:
    """Check if text contains any signal words indicating meaningful content."""
    if not text or not text.strip():
        return False
    lower = text.lower()
    return any(word in lower for word in SIGNAL_WORDS)


def _is_noise(text: str) -> bool:
    """Check if text is pure noise."""
    if not text or not text.strip():
        return True

    normalized = _normalize(text)

    # Match exact known noise phrases
    if normalized in NOISE_PHRASES:
        return True

    # Check for short filler words alone
    if normalized in FILLER_WORDS:
        return True

    # Very short message (under 2 words, 10 chars)
    words = normalized.split()
    if len(words) <= 2 and len(normalized) <= 10:
        return True

    # Laughter/slang patterns
    if LAUGHTER_PATTERN.match(normalized):
        return True

    return False


def _is_declarative(text: str) -> bool:
    """Check if text is a declarative sentence."""
    if not text or not text.strip():
        return False

    text = text.strip()

    # Check it starts with a subject
    starts_with_subject = bool(DECLARATIVE_START.match(text))

    # No question marks
    no_questions = "?" not in text

    # Ends with strong punctuation
    ends_with_punct = text.endswith(".") or text.endswith("!")

    # Low clause complexity (under 2 commas)
    low_complexity = text.count(",") < 2

    return starts_with_subject and no_questions and ends_with_punct and low_complexity


def is_noise(text: str, max_length: int = 80) -> bool:
    """Check if a message is noise and should be filtered from memory.

    Port of NoiseCheck.Check():
    Returns True if the text is too short AND not noise AND is declarative.
    (The C# logic: short + not-noise + declarative = skip it)

    For memory filtering, we return True if the message should be SKIPPED.
    """
    if not text or not text.strip():
        return True

    # Short messages that are noise phrases get filtered
    if _is_noise(text):
        return True

    # Very short messages need signal words to be kept
    if len(text) < max_length and not contains_signal_words(text):
        return True

    return False


def should_skip_memory(text: str, max_length: int = 80) -> bool:
    """Determine if a message should be skipped for memory processing.

    Combines noise check with signal word detection.
    """
    if _is_noise(text):
        return True

    if len(text) < max_length and not contains_signal_words(text):
        return True

    # Additional: check word count
    words = text.split()
    if len(words) < 3:
        return True

    return False
