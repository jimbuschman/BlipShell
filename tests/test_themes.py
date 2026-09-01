"""Theme-diversity metric (blipshell/memory/themes.py).

Ported from Wisp with its hard-won calibration intact; these tests pin the
properties that made the metric trustworthy there:

* rewordings of one sentence are ONE theme (a distinct-string count would
  score rumination as health);
* `domination` is computed WITHOUT chaining — single-link's largest family
  is a runaway blob on a paraphrase chain (measured 0.596 vs ~0.19 by hand);
* the prefix-filtered family join returns exactly what the exhaustive scan
  would.
"""

from blipshell.memory.themes import (
    NEAR_DUPLICATE_JACCARD,
    content_words,
    family_sizes,
    jaccard,
    signatures_of,
    theme_clusters,
    theme_diversity,
    ubiquitous_words,
)

REWORDED = [
    "The concept of unmeasured weight in silence invites us to reevaluate presence.",
    "The notion of unmeasured weight in silence invites us to reconsider presence.",
    "Unmeasured weight in silence asks us to reevaluate the concept of presence.",
]

DISTINCT = [
    "The sparrows returned to the feeder after the rain stopped this morning.",
    "Database migrations should archive rows instead of deleting them outright.",
    "I wonder whether the garden heron hunts differently in colder weather.",
]


def test_content_words_drops_stopwords_and_short_tokens():
    words = content_words("The birds ARE at the feeder, and I saw it")
    assert "the" not in words
    assert "and" not in words
    assert "birds" in words
    assert "feeder" in words


def test_jaccard_empty_sets_score_zero():
    assert jaccard(set(), {"bird"}) == 0.0
    assert jaccard(set(), set()) == 0.0


def test_rewordings_cluster_as_one_theme():
    clusters = theme_clusters(REWORDED)
    assert len(clusters) == 1


def test_distinct_subjects_stay_distinct():
    clusters = theme_clusters(DISTINCT)
    assert len(clusters) == 3


def test_domination_uses_no_chain_family():
    # A gradual paraphrase chain: consecutive texts overlap heavily, the ends
    # share almost nothing. Single-link chains it into one family; the
    # representative (no-chain) linkage must not.
    chain = [
        "alpha beta gamma delta epsilon zeta ",
        "beta gamma delta epsilon zeta eta",
        "gamma delta epsilon zeta eta theta",
        "delta epsilon zeta eta theta iota",
        "epsilon zeta eta theta iota kappa",
        "zeta eta theta iota kappa lam",
        "eta theta iota kappa lam mu",
        "theta iota kappa lam mu nu",
    ]
    stats = theme_diversity(chain)
    # Single-link follows the chain into fewer, bigger blobs...
    assert stats["domination_single_link"] > stats["domination"]
    # ...and the quoted figure is the unchained one.
    _, sizes = family_sizes(chain, link="representative")
    assert stats["largest_family"] == max(sizes)


def test_family_sizes_single_vs_representative():
    texts = REWORDED + DISTINCT
    _, single = family_sizes(texts, link="single")
    _, rep = family_sizes(texts, link="representative")
    # Both linkages agree here (rewordings are all near the FIRST member),
    # and neither merges the distinct subjects in.
    assert max(single) == 3
    assert max(rep) == 3
    assert len(single) == 4
    assert len(rep) == 4


def test_family_prefix_filter_matches_exhaustive_scan():
    # The prefix-filtered join is an optimization, not an approximation:
    # verify against a brute-force representative-linkage scan.
    texts = REWORDED + DISTINCT + REWORDED[:1] + [
        "Database migrations must archive rows rather than delete them outright.",
    ]
    family_of, _ = family_sizes(texts, link="representative")

    sigs = [content_words(t) for t in texts]
    brute: list[int] = []
    firsts: list[int] = []  # index of each family's first member
    for i, sig in enumerate(sigs):
        placed = None
        for fam, first in enumerate(firsts):
            if jaccard(sig, sigs[first]) >= NEAR_DUPLICATE_JACCARD:
                placed = fam
                break
        if placed is None:
            firsts.append(i)
            brute.append(len(firsts) - 1)
        else:
            brute.append(placed)
    assert family_of == brute


def test_ubiquitous_words_discounted_only_on_request():
    # Every text opens with the same tic; with the discount the tic no longer
    # counts as shared theme.
    tic = [
        "I am turning over the sparrows at the feeder near the window",
        "I am turning over whether migrations should archive old rows",
        "I am turning over the heron hunting in the cold morning river",
        "I am turning over the shape of the nightly maintenance schedule",
    ]
    plain = signatures_of(tic)
    discounted = signatures_of(tic, drop_ubiquitous=True)
    assert all("turning" in s for s in plain)
    assert all("turning" not in s for s in discounted)
    # Fewer than 3 texts: too few to tell a tic from a theme.
    assert ubiquitous_words(plain[:2]) == set()


def test_theme_diversity_empty_window():
    stats = theme_diversity([])
    assert stats["texts"] == 0
    assert stats["distinct_themes"] == 0
    assert stats["domination"] == 0.0


def test_theme_diversity_reports_both_linkages():
    stats = theme_diversity(REWORDED + DISTINCT)
    assert stats["texts"] == 6
    assert stats["distinct_themes"] == 4
    assert stats["largest_family"] == 3
    assert stats["domination"] == 0.5
    assert stats["domination_single_link"] == 0.5
