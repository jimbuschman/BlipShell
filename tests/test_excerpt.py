"""Query-relevant excerpts (V3 B3): the matched sentence survives the cut."""

from blipshell.memory.excerpt import excerpt, query_terms, split_sentences

FILLER = ("We talked about the weekend and the weather turning colder and a podcast "
          "about typewriters and whether the grinder needs replacing. ")


def _long(prefix_chars: int, fact: str, suffix_chars: int = 0) -> str:
    pre = (FILLER * 40)[:prefix_chars]
    suf = (FILLER * 40)[:suffix_chars]
    return f"{pre} {fact} {suf}".strip()


class TestQueryTerms:
    def test_drops_stopwords_and_possessives(self):
        assert query_terms("What is the Raspberry Pi's static IP address?") == \
            ["raspberry", "pi", "static", "ip", "address"]

    def test_keeps_numbers_and_dotted_tokens(self):
        assert "0.92" in query_terms("did we set 0.92 for consolidation")
        assert "192.168.4.77" in query_terms("is it 192.168.4.77?")


class TestExcerpt:
    def test_short_text_is_untouched(self):
        assert excerpt("My cat is Luna.", "cat name", max_chars=1200) == "My cat is Luna."

    def test_buried_fact_survives(self):
        text = _long(1300, "Oh and for the record: the Raspberry Pi's static IP is 192.168.4.77.")
        out = excerpt(text, "What is the Raspberry Pi's static IP address?", max_chars=1200)
        assert "192.168.4.77" in out
        assert len(out) <= 1200
        assert out.startswith("...")

    def test_fact_in_the_middle_gets_neighbours_and_both_ellipses(self):
        text = _long(1500, "The consolidation threshold is 0.92 and must not be lowered.", 1500)
        out = excerpt(text, "what threshold did we set for consolidation", max_chars=400)
        assert "0.92" in out and out.startswith("...") and out.endswith("...")
        assert len(out) <= 400
        assert out.count("grinder") >= 1, "neighbouring sentences are kept while they fit"

    def test_no_lexical_match_falls_back_to_prefix(self):
        text = _long(1300, "The Raspberry Pi's static IP is 192.168.4.77.")
        out = excerpt(text, "favourite colour", max_chars=300)
        assert out == text[:300] + "..."

    def test_prefers_the_sentence_with_most_terms(self):
        text = ("A" * 700 + ". The pi is on the shelf. " + "B" * 700 +
                ". The Raspberry Pi's static IP address is 10.0.0.5. " + "C" * 700)
        out = excerpt(text, "Raspberry Pi static IP address", max_chars=200)
        assert "10.0.0.5" in out

    def test_single_oversized_sentence_windows_around_the_hit(self):
        text = "x" * 3000 + " the answer is 42 " + "y" * 3000
        out = excerpt(text, "what is the answer", max_chars=200)
        assert "42" in out and len(out) <= 200

    def test_none_and_empty(self):
        assert excerpt(None, "q") == ""
        assert excerpt("", "q") == ""

    def test_split_sentences_handles_newlines(self):
        assert split_sentences("one. two\nthree") == ["one.", "two", "three"]
