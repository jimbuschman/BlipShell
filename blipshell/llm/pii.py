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
        # Suppress noisy warnings about non-English recognizers (es, it credit cards)
        # being skipped — we only use English, this is expected behavior
        presidio_logger = logging.getLogger("presidio-analyzer")
        prev_level = presidio_logger.level
        presidio_logger.setLevel(logging.ERROR)
        _presidio_analyzer = AnalyzerEngine()
        presidio_logger.setLevel(prev_level)
        _presidio_available = True
        logger.info("Presidio PII analyzer loaded (spaCy NER + regex)")
        return _presidio_analyzer
    except Exception as e:
        _presidio_available = False
        logger.info("Presidio not available, using regex-only PII sanitization: %s", e)
        return None


# Presidio entity type → placeholder mapping.
# Anything NOT listed here collapses to a single "[PII]" token, which makes a
# date, a passport number and a tax ID textually identical — so any type worth
# distinguishing has to be named. DATE_TIME especially: it's emitted by the
# default English recognizer set on almost every transcript, and losing
# temporal ordering ("we tried X Tuesday, it failed, Thursday we switched")
# destroys the causal spine that lesson extraction depends on.
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
    "DATE_TIME": "[DATE]",
}

# Entity types that get numbered, per-call-stable pseudonyms instead of one
# shared token. "[PERSON] asked [PERSON] to review [PERSON]'s PR" destroys
# coreference and agency — and lesson extraction is specifically asked for
# concrete detail about who did what and what followed. Numbering restores the
# distinctions at no privacy cost: the real names are still gone.
_NUMBERED_ENTITY_TYPES = frozenset({"PERSON"})

# IPs to skip — Presidio catches all IPs so we need to whitelist local ranges.
# 172.16/12 included: that's the Docker bridge range, and scrubbing it turns
# container debugging transcripts into mush.
_LOCAL_IP_PATTERN = re.compile(
    r'^(?:127\.0\.0\.1|0\.0\.0\.0'
    r'|192\.168\.\d{1,3}\.\d{1,3}'
    r'|10\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    r'|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})$'
)


def _sanitize_with_presidio(text: str) -> str:
    """Sanitize using Presidio NER engine."""
    analyzer = _get_presidio_analyzer()
    if analyzer is None:
        return _sanitize_with_regex(text)

    # Run the API-key regexes FIRST, then analyze the rewritten text.
    # Presidio's offsets are computed against whatever string it analyzed, so
    # rewriting the string afterwards invalidated every offset past the first
    # substitution (ghp_ + 36 chars -> "[API_KEY]" shifts by -31). Later
    # entities were then sliced at the wrong position — mangling innocent words
    # while potentially leaving the actual PII in place. A pasted token is
    # exactly what a debugging transcript contains, so this fired on the sessions
    # that matter most (deep-dive 2026-08-04).
    result = text
    for p in _API_KEY_PATTERNS:
        result = p.pattern.sub(p.replacement, result)

    results = analyzer.analyze(text=result, language="en")

    # Assign pseudonym numbers in FORWARD reading order, so [PERSON_1] is the
    # first person mentioned. Application below happens in reverse (to keep
    # offsets valid), which would otherwise number them backwards.
    pseudonyms: dict[tuple[str, str], str] = {}
    for r in sorted(results, key=lambda x: x.start):
        if r.entity_type not in _NUMBERED_ENTITY_TYPES:
            continue
        key = (r.entity_type, result[r.start:r.end].strip().lower())
        if key not in pseudonyms:
            base = _PRESIDIO_PLACEHOLDERS.get(r.entity_type, "[PII]").strip("[]")
            same_type = sum(1 for k in pseudonyms if k[0] == r.entity_type)
            pseudonyms[key] = f"[{base}_{same_type + 1}]"

    # Apply Presidio results in reverse order (so offsets stay valid).
    # Spans can OVERLAP (a URL span across an EMAIL span, etc.); applying
    # both at their raw offsets would slice the second at positions the first
    # already shifted — mangled text, and potentially the actual PII left in
    # place. Track how far left the applied replacements reach and TRUNCATE
    # any span that crosses it to the still-unreplaced region: skipping it
    # outright would leave the non-overlapping remainder of the PII behind
    # ("john@" from an EMAIL span half-covered by a URL span).
    applied_start = len(result) + 1
    for r in sorted(results, key=lambda x: x.start, reverse=True):
        end = min(r.end, applied_start)
        if end <= r.start:
            continue    # fully covered by an already-applied span
        original = result[r.start:end]

        # Skip local/LAN IPs
        if r.entity_type == "IP_ADDRESS" and _LOCAL_IP_PATTERN.match(original):
            continue

        # Skip URLs that look like localhost endpoints
        if r.entity_type == "URL" and ("localhost" in original or "127.0.0.1" in original):
            continue

        key = (r.entity_type, original.strip().lower())
        placeholder = pseudonyms.get(
            key, _PRESIDIO_PLACEHOLDERS.get(r.entity_type, "[PII]")
        )
        result = result[:r.start] + placeholder + result[end:]
        applied_start = r.start

    return result


# ---------------------------------------------------------------------------
# Regex fallback engine
# ---------------------------------------------------------------------------

@dataclass
class PIIPattern:
    name: str
    pattern: re.Pattern
    replacement: str


# API key patterns — used by both engines (Presidio doesn't detect these).
# Ordered most-specific first. These are the credentials most likely to be
# pasted into a debugging session; a leaked token is the one PII category with
# an immediate, concrete cost, so breadth here matters more than elsewhere.
_API_KEY_PATTERNS = [
    # GitHub: ghp_ (classic PAT), gho_/ghu_/ghs_ (OAuth/user/server),
    # github_pat_ (fine-grained). Only ghp_ was covered before.
    PIIPattern("github_token", re.compile(r'\bgh[pousr]_[a-zA-Z0-9]{36,}\b'), "[API_KEY]"),
    PIIPattern("github_pat", re.compile(r'\bgithub_pat_[a-zA-Z0-9_]{20,}\b'), "[API_KEY]"),
    PIIPattern("aws_key", re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b'), "[API_KEY]"),
    PIIPattern("slack_token", re.compile(r'\bxox[baprs]-[a-zA-Z0-9-]{10,}\b'), "[API_KEY]"),
    PIIPattern("google_api_key", re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b'), "[API_KEY]"),
    # Private key blocks — replace the whole armored body, not just the header.
    PIIPattern("private_key", re.compile(
        r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----',
        re.DOTALL,
    ), "[PRIVATE_KEY]"),
    PIIPattern("bearer_header", re.compile(
        r'\b[Bb]earer\s+[A-Za-z0-9._~+/-]{20,}=*'
    ), "Bearer [API_KEY]"),
    # Connection-string passwords. Two shapes cover the real cases:
    #
    # URI userinfo — postgres://admin:s3cret@host/db, mongodb+srv://...,
    # http://user:pass@host (basic auth). Only the password is replaced;
    # scheme, user, host and database are exactly what a debugging session
    # needs and are kept. URLs WITHOUT userinfo (http://localhost:11434)
    # have no ":password@" and never match.
    PIIPattern("uri_password", re.compile(
        r'([a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@]+:)[^\s@]+(?=@)'
    ), r"\1[PASSWORD]"),
    # key=value — ADO/ODBC strings (Password=s3cret;) and .env lines
    # (DB_PASSWORD=hunter2). The = must be UNSPACED: `password = value` is
    # how code reads (PEP 8), and scrubbing code discussions is the
    # skeletonAnimationControl lesson again; `password=value` is how
    # connection strings and env files read.
    PIIPattern("keyval_password", re.compile(
        r'(?i)\b([a-z0-9_]*(?:password|passwd|pwd))=([^;\s"\x27]+)'
    ), r"\1=[PASSWORD]"),
    PIIPattern("jwt", re.compile(
        r'\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b'
    ), "[JWT]"),
    # sk-/pk-/api- style. Two guards against eating ordinary identifiers,
    # because this pattern runs on EVERY interactive turn via
    # sanitize_secrets: the separator is REQUIRED (every real issuer format
    # has one — sk-proj-, sk-ant-, sk_live_, pk_test_ — but camelCase words
    # don't: skeletonAnimationControl is `sk` + 22 valid chars and was being
    # replaced with [API_KEY] mid-conversation), and the tail must contain a
    # digit (real keys are high-entropy; api_configuration_manager is not).
    PIIPattern("generic_api_key", re.compile(
        r'\b(?:sk|pk|api)[_-](?=[a-zA-Z_-]*\d)[a-zA-Z0-9_-]{16,}\b'
    ), "[API_KEY]"),
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

    FULL sanitization — secrets AND identity (names, dates, locations, URLs).
    Appropriate for background/bulk calls where the text is processed once and
    the output is stored. NOT appropriate for interactive chat: see
    sanitize_secrets() for why.

    Returns the sanitized text. If no PII found, returns original string unchanged.
    """
    if not text:
        return text

    result = _sanitize_with_presidio(text)

    if result != text:
        logger.debug("PII sanitized before cloud routing")
    return result


def sanitize_secrets(text: str) -> str:
    """Strip credentials only — no NER, no identity redaction.

    This is what the INTERACTIVE path uses on cloud endpoints, because the two
    PII categories have opposite cost profiles:

    - Secrets (tokens, keys, JWTs, private keys) cost the model NOTHING in
      comprehension — it never needed the real value to answer — and they're
      the one category whose leakage has an immediate, concrete cost. Always
      worth removing.
    - Identity (names, dates, locations, URLs, orgs) is where redaction gets
      expensive and where it protects least. spaCy NER tags product names as
      entities, so a technical conversation loses "Groq", "Ollama",
      "Presidio", "Devstral" as PERSON/LOCATION/ORG — and it leaks anyway,
      since no pattern catches a username embedded in every file path
      (C:\\Users\\<name>\\...), nicknames, handles, or names inside
      identifiers. Full redaction on this path would pay the whole
      comprehension cost for partial protection.

    Identity protection belongs at the ROUTING layer instead — keep a
    conversation local — which is a boundary that can actually be reasoned
    about. See docs/V2_PLAN.md (D1).

    Regex-only, so it's fast enough for a per-turn interactive path (full
    sanitization would run spaCy NER over the whole assembled context, recall
    pool included, on every turn).
    """
    if not text:
        return text
    result = text
    for p in _API_KEY_PATTERNS:
        result = p.pattern.sub(p.replacement, result)
    if result != text:
        logger.info("Credentials redacted before cloud routing")
    return result


def sanitize_messages_secrets(messages: list[dict]) -> list[dict]:
    """Copy of `messages` with credentials stripped from every text field.

    Returns a NEW list with new dicts — the caller's conversation history must
    not be mutated, or the redaction would be written back into the session
    and the stored memory.
    """
    out: list[dict] = []
    for msg in messages:
        new_msg = dict(msg)
        content = new_msg.get("content")
        if isinstance(content, str):
            new_msg["content"] = sanitize_secrets(content)
        out.append(new_msg)
    return out


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
