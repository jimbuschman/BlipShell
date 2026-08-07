"""Tests for PII detection and sanitization.

Tests run against whichever engine is available (Presidio or regex fallback).
Both engines should produce equivalent results for structured PII patterns.
Presidio-only tests (names, addresses) are skipped when Presidio isn't installed.
"""

import pytest
from blipshell.llm.pii import (
    sanitize_text, has_pii, is_presidio_available,
    _sanitize_with_regex,
)


class TestSanitizeText:
    """Tests for sanitize_text() — works with either engine."""

    def test_empty_string(self):
        assert sanitize_text("") == ""

    def test_none_passthrough(self):
        assert sanitize_text(None) is None

    def test_no_pii(self):
        text = "The weather is nice today and I like Python."
        assert sanitize_text(text) == text

    # --- Email ---
    def test_email(self):
        result = sanitize_text("Contact me at john@example.com please")
        assert "john@example.com" not in result
        assert "[EMAIL]" in result

    def test_email_with_plus(self):
        result = sanitize_text("Email: user+tag@domain.co.uk")
        assert "user+tag@domain.co.uk" not in result

    # --- Phone ---
    def test_phone_us_dashes(self):
        result = sanitize_text("Call 555-123-4567")
        assert "555-123-4567" not in result

    def test_phone_us_parens(self):
        result = sanitize_text("Call (555) 123-4567")
        assert "123-4567" not in result

    def test_phone_us_with_country(self):
        result = sanitize_text("Call +1-555-123-4567")
        assert "123-4567" not in result

    # --- SSN ---
    def test_ssn(self):
        result = sanitize_text("SSN: 123-45-6789")
        assert "123-45-6789" not in result

    # --- Credit card ---
    def test_credit_card_spaces(self):
        result = sanitize_text("Card: 4111 1111 1111 1111")
        assert "4111" not in result

    def test_credit_card_dashes(self):
        result = sanitize_text("Card: 4111-1111-1111-1111")
        assert "4111" not in result

    def test_credit_card_no_sep(self):
        result = sanitize_text("Card: 4111111111111111")
        assert "4111111111111111" not in result

    # --- API keys ---
    def test_github_token(self):
        result = sanitize_text("Token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij")
        assert "[API_KEY]" in result

    def test_aws_key(self):
        result = sanitize_text("Key: AKIAIOSFODNN7EXAMPLE")
        assert "[API_KEY]" in result

    def test_generic_api_key(self):
        result = sanitize_text("Key: sk_live_12345678901234567890")
        assert "[API_KEY]" in result

    # --- IP address ---
    def test_ip_address(self):
        result = sanitize_text("Server at 203.0.113.42")
        assert "203.0.113.42" not in result

    def test_ip_localhost_preserved(self):
        text = "Ollama at 127.0.0.1:11434"
        assert "127.0.0.1" in sanitize_text(text)

    def test_ip_lan_preserved(self):
        text = "Server at 192.168.1.100"
        assert "192.168.1.100" in sanitize_text(text)

    def test_ip_10_network_preserved(self):
        text = "Server at 10.0.0.1"
        assert "10.0.0.1" in sanitize_text(text)

    # --- Multiple PII in one string ---
    def test_multiple_pii(self):
        text = "Email john@test.com, SSN 123-45-6789, call 555-867-5309"
        result = sanitize_text(text)
        assert "john@test.com" not in result
        assert "123-45-6789" not in result
        assert "555-867-5309" not in result


class TestRegexFallback:
    """Tests specifically for the regex engine (always available)."""

    def test_email(self):
        assert _sanitize_with_regex("Contact john@example.com") == "Contact [EMAIL]"

    def test_phone(self):
        assert _sanitize_with_regex("Call 555-123-4567") == "Call [PHONE]"

    def test_ssn(self):
        assert _sanitize_with_regex("SSN: 123-45-6789") == "SSN: [SSN]"

    def test_api_key(self):
        assert _sanitize_with_regex("Key: AKIAIOSFODNN7EXAMPLE") == "Key: [API_KEY]"


class TestPresidioEngine:
    """Tests for Presidio-specific detection (names, addresses).

    Skipped when Presidio is not installed.
    """

    @pytest.fixture(autouse=True)
    def _require_presidio(self):
        if not is_presidio_available():
            pytest.skip("Presidio not installed")

    def test_detects_person_name(self):
        result = sanitize_text("My name is John Smith and I live in Denver")
        assert "John Smith" not in result

    def test_detects_location(self):
        result = sanitize_text("I moved to San Francisco last year")
        assert "San Francisco" not in result

    def test_preserves_non_pii(self):
        text = "Python is a programming language"
        assert sanitize_text(text) == text


class TestHasPII:
    """Tests for has_pii() — always uses regex (fast path)."""

    def test_empty_string(self):
        assert has_pii("") is False

    def test_no_pii(self):
        assert has_pii("Just a normal sentence.") is False

    def test_has_email(self):
        assert has_pii("Contact john@example.com") is True

    def test_has_phone(self):
        assert has_pii("Call 555-123-4567") is True

    def test_has_ssn(self):
        assert has_pii("SSN 123-45-6789") is True

    def test_localhost_not_pii(self):
        assert has_pii("Server at 127.0.0.1") is False

    def test_lan_not_pii(self):
        assert has_pii("Server at 192.168.0.225") is False


# ---------------------------------------------------------------------------
# Presidio offset handling + pseudonyms, tested with a STUB analyzer.
#
# The offset bug is pure index arithmetic, so it's testable without Presidio
# installed: the API-key regexes used to rewrite the string AFTER Presidio had
# computed offsets against the original, so every entity past the first key
# match was sliced at the wrong position (deep-dive 2026-08-04).
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from blipshell.llm import pii as pii_mod


@dataclass
class _StubResult:
    entity_type: str
    start: int
    end: int


class _StubAnalyzer:
    """Finds fixed substrings and reports real offsets into whatever it was
    given — which is exactly how Presidio behaves."""

    def __init__(self, wanted: dict):
        self.wanted = wanted          # {substring: entity_type}
        self.analyzed_text = None

    def analyze(self, text: str, language: str = "en"):
        self.analyzed_text = text
        out = []
        for needle, etype in self.wanted.items():
            start = 0
            while (idx := text.find(needle, start)) != -1:
                out.append(_StubResult(etype, idx, idx + len(needle)))
                start = idx + len(needle)
        return out


@pytest.fixture
def stub_presidio(monkeypatch):
    def _install(wanted):
        stub = _StubAnalyzer(wanted)
        monkeypatch.setattr(pii_mod, "_get_presidio_analyzer", lambda: stub)
        return stub
    return _install


class TestPresidioOffsetCorrectness:
    def test_api_key_before_name_does_not_corrupt_the_name(self, stub_presidio):
        """The regression: a pasted token earlier in the text shifted every
        later offset, so the name was sliced at the wrong position."""
        stub = stub_presidio({"Kortney": "PERSON"})
        text = "token ghp_" + "a" * 36 + " belongs to Kortney here"

        result = pii_mod.sanitize_text(text)

        assert "[API_KEY]" in result
        assert "Kortney" not in result, "the real name survived the scrub"
        assert "[PERSON" in result
        # Nothing around the name got eaten
        assert result.endswith(" here")
        assert "belongs to" in result

    def test_analyzer_sees_the_post_regex_text(self, stub_presidio):
        """Offsets are only valid if the analyzer analyzed the same string we
        then mutate."""
        stub = stub_presidio({"Kortney": "PERSON"})
        pii_mod.sanitize_text("key ghp_" + "b" * 36 + " and Kortney")
        assert "[API_KEY]" in stub.analyzed_text
        assert "ghp_" not in stub.analyzed_text

    def test_multiple_keys_and_multiple_entities(self, stub_presidio):
        stub_presidio({"Alice": "PERSON", "Bob": "PERSON"})
        text = (
            "AKIAABCDEFGHIJKLMNOP then Alice, then ghp_" + "c" * 36 + ", then Bob."
        )
        result = pii_mod.sanitize_text(text)
        assert "Alice" not in result and "Bob" not in result
        assert "AKIAABCDEFGHIJKLMNOP" not in result
        assert result.endswith(".")


class TestPersonPseudonyms:
    def test_distinct_people_get_distinct_tokens(self, stub_presidio):
        """One shared [PERSON] token destroyed coreference: "[PERSON] asked
        [PERSON] to review [PERSON]'s PR" is unusable for lesson extraction."""
        stub_presidio({"Alice": "PERSON", "Bob": "PERSON"})
        result = pii_mod.sanitize_text("Alice asked Bob to review Alice's PR")

        assert "[PERSON_1]" in result
        assert "[PERSON_2]" in result
        # Same person -> same token, so "who did what" survives
        assert result.count("[PERSON_1]") == 2

    def test_numbering_follows_reading_order(self, stub_presidio):
        stub_presidio({"Zach": "PERSON", "Amy": "PERSON"})
        result = pii_mod.sanitize_text("Zach spoke first, then Amy replied")
        assert result.index("[PERSON_1]") < result.index("[PERSON_2]")
        assert "Zach" not in result and "Amy" not in result

    def test_pseudonyms_are_case_insensitive_for_the_same_person(self, stub_presidio):
        stub_presidio({"Alice": "PERSON", "alice": "PERSON"})
        result = pii_mod.sanitize_text("Alice and alice are one person")
        assert "[PERSON_2]" not in result


class TestUnmappedEntityTypes:
    def test_dates_get_a_date_token_not_generic_pii(self, stub_presidio):
        """DATE_TIME used to fall through to [PII], making a date textually
        identical to a passport number and destroying event ordering."""
        stub_presidio({"Tuesday": "DATE_TIME"})
        result = pii_mod.sanitize_text("we shipped it on Tuesday")
        assert "[DATE]" in result
        assert "[PII]" not in result

    def test_docker_bridge_ip_is_preserved(self, stub_presidio):
        stub_presidio({"172.17.0.2": "IP_ADDRESS"})
        result = pii_mod.sanitize_text("container at 172.17.0.2 refused")
        assert "172.17.0.2" in result, "Docker-bridge IP should not be scrubbed"

    def test_public_ip_still_scrubbed(self, stub_presidio):
        stub_presidio({"8.8.8.8": "IP_ADDRESS"})
        result = pii_mod.sanitize_text("resolver at 8.8.8.8 responded")
        assert "8.8.8.8" not in result


class TestExpandedKeyPatterns:
    @pytest.mark.parametrize("secret", [
        "gho_" + "d" * 36,
        "github_pat_" + "e" * 30,
        "xoxb-1234567890-abcdefghij",
        "AIza" + "f" * 35,
        "ASIAABCDEFGHIJKLMNOP",
    ])
    def test_common_token_formats_are_scrubbed(self, secret):
        result = sanitize_text(f"the secret is {secret} ok")
        assert secret not in result

    def test_jwt_scrubbed(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p"
        assert jwt not in sanitize_text(f"Authorization: {jwt}")

    def test_private_key_block_scrubbed(self):
        blob = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA1234567890\nabcdefgh\n"
            "-----END RSA PRIVATE KEY-----"
        )
        result = sanitize_text(f"here it is:\n{blob}\ndone")
        assert "MIIEowIBAAKCAQEA1234567890" not in result
        assert "[PRIVATE_KEY]" in result
        assert result.endswith("done")


class TestSecretsOnlySanitization:
    """The interactive path strips credentials but NOT identity.

    Redacting a token costs the model nothing (it never needed the real value)
    and is the one category with immediate concrete harm. Redacting identity
    would cost a lot and protect little: spaCy tags product names as entities,
    and a username leaks through every file path regardless. Identity
    protection belongs at the routing layer (keep it local) — see V2_PLAN D1.
    """

    def test_api_key_is_removed(self):
        secret = "ghp_" + "a" * 36
        result = pii_mod.sanitize_secrets(f"my token is {secret} ok")
        assert secret not in result
        assert "[API_KEY]" in result

    def test_names_are_preserved(self):
        text = "Kortney and I discussed the Ollama gate on Tuesday"
        assert pii_mod.sanitize_secrets(text) == text

    def test_technical_nouns_survive(self):
        """The specific failure full sanitization would cause on this path."""
        text = "Groq rate-limited, so Presidio ran locally against Devstral"
        assert pii_mod.sanitize_secrets(text) == text

    def test_urls_and_paths_preserved(self):
        text = r"see https://docs.example.com and C:\Users\[user]\app.py"
        assert pii_mod.sanitize_secrets(text) == text

    def test_empty_and_none_safe(self):
        assert pii_mod.sanitize_secrets("") == ""
        assert pii_mod.sanitize_secrets(None) is None


class TestMessageSanitizationIsNonDestructive:
    def test_returns_a_copy_and_never_mutates_the_caller(self):
        """The load-bearing property: `messages` becomes conversation history
        and gets persisted as memory. If the transform mutated it, the
        redaction would be written back into the session and the store."""
        secret = "ghp_" + "b" * 36
        original = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": f"deploy with {secret}"},
        ]
        snapshot = [dict(m) for m in original]

        cleaned = pii_mod.sanitize_messages_secrets(original)

        assert original == snapshot, "caller's messages were mutated"
        assert original[1]["content"] == f"deploy with {secret}"
        assert secret not in cleaned[1]["content"]
        assert cleaned is not original
        assert cleaned[1] is not original[1]

    def test_non_string_content_passes_through(self):
        """Tool-call messages and image payloads must survive untouched."""
        messages = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "read_file"}}]},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        cleaned = pii_mod.sanitize_messages_secrets(messages)
        assert cleaned[0]["content"] is None
        assert cleaned[0]["tool_calls"] == messages[0]["tool_calls"]
        assert cleaned[1]["content"] == messages[1]["content"]

    def test_all_roles_are_scrubbed(self):
        """Secrets can appear in a tool result, not just a user message."""
        secret = "AKIAABCDEFGHIJKLMNOP"
        messages = [{"role": "tool", "content": f"env: AWS_KEY={secret}"}]
        cleaned = pii_mod.sanitize_messages_secrets(messages)
        assert secret not in cleaned[0]["content"]


class TestGenericKeyPatternPrecision:
    """generic_api_key runs on EVERY interactive turn via sanitize_secrets.
    With an optional separator it ate ordinary camelCase identifiers —
    skeletonAnimationControl is `sk` + 22 valid chars — replacing them with
    [API_KEY] mid-conversation before the model saw them."""

    @pytest.mark.parametrize("identifier", [
        "skeletonAnimationControl",
        "apiConfigurationManagerClass",
        "pkgResolverInitialization",
        "api_configuration_manager",   # separator present, but no digit
    ])
    def test_ordinary_identifiers_survive(self, identifier):
        text = f"the {identifier} loads first"
        assert pii_mod.sanitize_secrets(text) == text

    # Shaped like real issuer formats (OpenAI sk-proj-, Stripe sk_live_ /
    # pk_test_, Anthropic sk-ant-) but with fabricated tails, assembled at
    # runtime — GitHub push protection rejects anything matching the genuine
    # token patterns anywhere in pushed history, verbatim doc-example keys
    # included.
    @pytest.mark.parametrize("key", [
        "sk-proj-" + "fake0" * 5,
        "sk_" + "live_" + "FAKE1FAKE2FAKE3FAKE4",
        "sk-ant-" + "api03-" + "FAKE5FAKE6FAKE7FAKE8",
        "pk_" + "test_" + "FAKE9FAKE0FAKE1FAKE2",
    ])
    def test_real_key_formats_are_still_caught(self, key):
        assert key not in pii_mod.sanitize_secrets(f"token {key} leaked")


class TestOverlappingPresidioSpans:
    def test_partial_overlap_does_not_mangle_text(self, stub_presidio):
        """Reverse-order replacement assumes disjoint spans. Presidio can
        return partial overlaps; slicing the second span at offsets the first
        already shifted mangled the text and could leave the PII in place."""
        # The left span must reach DEEPER into the right span than the
        # replacement placeholder is wide — a shallower overlap coincidentally
        # slices off exactly the placeholder and produces clean-looking output
        # (which is how the first version of this test failed to catch the
        # bug). Here EMAIL [5..21) reaches 11 chars into URL [10..28); at raw
        # offsets the stale slice eats "TAILM" from the tail.
        stub = stub_presidio({
            "john@x.com/veryl": "EMAIL_ADDRESS",
            "x.com/verylongpath": "URL",
        })
        text = "mail john@x.com/verylongpath TAILMARK"

        result = pii_mod.sanitize_text(text)

        assert "john" not in result, "the PII survived the overlap"
        assert "TAILMARK" in result, f"text after the spans was mangled: {result!r}"
        assert result.startswith("mail ")

    def test_disjoint_spans_are_all_applied(self, stub_presidio):
        """The overlap guard must not skip spans that merely sit close."""
        stub = stub_presidio({"Kortney": "PERSON", "Dallas": "LOCATION"})
        result = pii_mod.sanitize_text("Kortney flew to Dallas")
        assert "Kortney" not in result
        assert "Dallas" not in result


class TestConnectionStringPasswords:
    """Credential shapes the key patterns missed: passwords embedded in
    connection strings. Scrubbed on EVERY cloud path, chat included — and the
    parts a debugging session actually needs (scheme, user, host, database)
    are kept."""

    @pytest.mark.parametrize("conn,kept", [
        ("postgres://admin:s3cret@db.internal:5432/prod", "db.internal:5432/prod"),
        ("mongodb+srv://app:Tr0ub4dor@cluster0.mongodb.net/store", "cluster0"),
        ("redis://default:hunter2@cache:6379/0", "cache:6379"),
        ("https://jim:basicauthpw@internal.example/api", "internal.example"),
    ])
    def test_uri_password_is_scrubbed_host_kept(self, conn, kept):
        out = pii_mod.sanitize_secrets(f"failing with {conn} here")
        assert "[PASSWORD]@" in out
        assert kept in out, "the host/db the debugging needs was lost"
        for secret in ("s3cret", "Tr0ub4dor", "hunter2", "basicauthpw"):
            assert secret not in out

    @pytest.mark.parametrize("text,secret", [
        ("Server=db;User Id=sa;Password=Sup3rS3cret;TrustCert=true", "Sup3rS3cret"),
        ("DB_PASSWORD=hunter2", "hunter2"),
        ("set PGPASSWORD=abc123xyz before running", "abc123xyz"),
        ("conn opts: pwd=letmein;timeout=30", "letmein"),
    ])
    def test_keyval_password_is_scrubbed(self, text, secret):
        out = pii_mod.sanitize_secrets(text)
        assert secret not in out
        assert "[PASSWORD]" in out

    def test_plain_urls_without_userinfo_are_untouched(self):
        text = "Ollama runs at http://localhost:11434 and the UI at https://example.com/app"
        assert pii_mod.sanitize_secrets(text) == text

    def test_spaced_assignment_in_code_is_untouched(self):
        """`password = get_password()` is code being DISCUSSED, not a secret —
        the unspaced-= requirement is what keeps code conversations legible."""
        text = "in main.py: password = os.environ.get('DB_PASS')"
        assert pii_mod.sanitize_secrets(text) == text

    def test_ado_string_keeps_the_rest_of_the_settings(self):
        out = pii_mod.sanitize_secrets("Server=db;Password=s3cret;Timeout=30")
        assert out == "Server=db;Password=[PASSWORD];Timeout=30"
