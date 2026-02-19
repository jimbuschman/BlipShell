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

# Filler words — single-word (noise only when used alone)
FILLER_WORDS = {"um", "uh", "well", "like"}

# Multi-word filler phrases (checked separately after normalization)
FILLER_PHRASES = {"you know", "i mean"}

# Laughter pattern
LAUGHTER_PATTERN = re.compile(r"^(ha|lol|lmao|rofl)+[!]*$", re.IGNORECASE)


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

    # Check for filler words/phrases alone
    if normalized in FILLER_WORDS or normalized in FILLER_PHRASES:
        return True

    # Very short message (under 2 words, 10 chars)
    words = normalized.split()
    if len(words) <= 2 and len(normalized) <= 10:
        return True

    # Laughter/slang patterns
    if LAUGHTER_PATTERN.match(normalized):
        return True

    return False


def should_skip_memory(text: str, max_length: int = 80, min_word_count: int = 3) -> bool:
    """Determine if a message should be skipped for memory processing.

    Combines noise check with signal word detection.

    Args:
        text: The message text.
        max_length: Messages shorter than this without signal words are skipped.
        min_word_count: Messages with fewer words than this are skipped.
            Defaults to 3; can be overridden via NoiseConfig.min_word_count.
    """
    if _is_noise(text):
        return True

    if len(text) < max_length and not contains_signal_words(text):
        return True

    # Check word count (configurable via min_word_count)
    words = text.split()
    if len(words) < min_word_count:
        return True

    return False
