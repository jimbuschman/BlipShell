"""PII detection and sanitization for cloud-bound prompts.

Uses Microsoft Presidio (spaCy NER + regex) when available for high-accuracy
detection of names, addresses, and context-dependent PII. Falls back to
regex-only pattern matching if Presidio is not installed.

Local Ollama calls are NOT sanitized — raw text is preserved for search quality.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Presidio engine (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_presidio_analyzer = None
_presidio_available = None  # None = not checked yet


def _get_presidio_analyzer():
    """Lazy-load Presidio analyzer. Returns None if not installed."""
    global _presidio_analyzer, _presidio_available

    if _presidio_available is False:
        return None
    if _presidio_analyzer is not None:
        return _presidio_analyzer

    try:
        from presidio_analyzer import AnalyzerEngine
        _presidio_analyzer = AnalyzerEngine()
        _presidio_available = True
        logger.info("Presidio PII analyzer loaded (spaCy NER + regex)")
        return _presidio_analyzer
    except Exception as e:
        _presidio_available = False
        logger.info("Presidio not available, using regex-only PII sanitization: %s", e)
        return None


# Presidio entity type → placeholder mapping
_PRESIDIO_PLACEHOLDERS = {
    "PERSON": "[PERSON]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE]",
    "US_SSN": "[SSN]",
    "CREDIT_CARD": "[CARD]",
    "IP_ADDRESS": "[IP]",
    "LOCATION": "[LOCATION]",
    "US_DRIVER_LICENSE": "[ID]",
    "US_PASSPORT": "[ID]",
    "US_BANK_NUMBER": "[BANK]",
    "IBAN_CODE": "[BANK]",
    "CRYPTO": "[CRYPTO]",
    "MEDICAL_LICENSE": "[ID]",
    "URL": "[URL]",
    "NRP": "[NRP]",  # nationality, religion, political group
}

# IPs to skip — Presidio catches all IPs so we need to whitelist local ranges
_LOCAL_IP_PATTERN = re.compile(
    r'^(?:127\.0\.0\.1|0\.0\.0\.0|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})$'
)


def _sanitize_with_presidio(text: str) -> str:
    """Sanitize using Presidio NER engine."""
    analyzer = _get_presidio_analyzer()
    if analyzer is None:
        return _sanitize_with_regex(text)

    results = analyzer.analyze(text=text, language="en")

    # Also run our regex patterns for API keys (Presidio doesn't detect these)
    result = text
    for p in _API_KEY_PATTERNS:
        result = p.pattern.sub(p.replacement, result)

    # Apply Presidio results in reverse order (so offsets stay valid)
    for r in sorted(results, key=lambda x: x.start, reverse=True):
        original = result[r.start:r.end]

        # Skip local/LAN IPs
        if r.entity_type == "IP_ADDRESS" and _LOCAL_IP_PATTERN.match(original):
            continue

        # Skip URLs that look like localhost endpoints
        if r.entity_type == "URL" and ("localhost" in original or "127.0.0.1" in original):
            continue

        placeholder = _PRESIDIO_PLACEHOLDERS.get(r.entity_type, "[PII]")
        result = result[:r.start] + placeholder + result[r.end:]

    return result


# ---------------------------------------------------------------------------
# Regex fallback engine
# ---------------------------------------------------------------------------

@dataclass
class PIIPattern:
    name: str
    pattern: re.Pattern
    replacement: str


# API key patterns — used by both engines (Presidio doesn't detect these)
_API_KEY_PATTERNS = [
    PIIPattern("github_token", re.compile(r'\bghp_[a-zA-Z0-9]{36}\b'), "[API_KEY]"),
    PIIPattern("aws_key", re.compile(r'\bAKIA[0-9A-Z]{16}\b'), "[API_KEY]"),
    PIIPattern("generic_api_key", re.compile(r'\b(?:sk|pk|api)[_-][a-zA-Z0-9_-]{20,}\b'), "[API_KEY]"),
]

# Pre-compiled patterns — ordered from most specific to least
PII_PATTERNS = [
    *_API_KEY_PATTERNS,
    # SSN (before phone, since format overlaps)
    PIIPattern("ssn", re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[SSN]"),
    # Credit card (16 digits with optional separators)
    PIIPattern("credit_card", re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'), "[CARD]"),
    # Email
    PIIPattern("email", re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), "[EMAIL]"),
    # Phone (US formats: 123-456-7890, (123) 456-7890, +1-123-456-7890)
    PIIPattern("phone", re.compile(r'(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), "[PHONE]"),
    # IP address (skip common localhost/LAN ranges used in configs)
    PIIPattern("ip_address", re.compile(
        r'\b(?!127\.0\.0\.1\b)(?!192\.168\.\d{1,3}\.\d{1,3}\b)(?!10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)'
        r'(?!0\.0\.0\.0\b)(?!localhost\b)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    ), "[IP]"),
]


def _sanitize_with_regex(text: str) -> str:
    """Regex-only fallback sanitization."""
    result = text
    for p in PII_PATTERNS:
        result = p.pattern.sub(p.replacement, result)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize_text(text: str) -> str:
    """Replace PII with placeholders. Uses Presidio if available, regex otherwise.

    Returns the sanitized text. If no PII found, returns original string unchanged.
    """
    if not text:
        return text

    result = _sanitize_with_presidio(text)

    if result != text:
        logger.debug("PII sanitized before cloud routing")
    return result


def has_pii(text: str) -> bool:
    """Quick check whether text contains any PII patterns.

    Uses regex only (fast path) — doesn't load Presidio for a quick check.
    """
    if not text:
        return False
    return any(p.pattern.search(text) for p in PII_PATTERNS)


def is_presidio_available() -> bool:
    """Check if Presidio is installed and working."""
    return _get_presidio_analyzer() is not None
