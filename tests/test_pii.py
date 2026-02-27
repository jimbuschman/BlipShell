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
