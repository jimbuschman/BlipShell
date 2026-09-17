"""Loading Presidio is a one-time, serialized, quiet event.

``AnalyzerEngine()`` logs a WARNING on the ``presidio-analyzer`` logger for
every predefined recognizer whose language the en-only registry rejects
(eleven lines at startup, seen live 2026-09-17). The loader suppressed them
by lowering that logger's level around the constructor - correct for one
caller, wrong for two: the agent's PII engine report and the memory worker's
first cloud-bound call both load Presidio at startup, on different threads,
with no lock. The first to finish restored the level while the second was
still constructing (its warnings leaked), the second then restored the
first's temporary ERROR level for good, and spaCy was loaded twice.

These tests fake ``presidio_analyzer`` with a slow constructor and pin: one
construction, one shared analyzer, the known noise dropped, anything else
Presidio says at WARNING kept, and the logger's level untouched afterwards.
"""

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest

from blipshell.llm import pii

PRESIDIO_LOGGER = "presidio-analyzer"
NOISE = ("Recognizer not added to registry because language is not supported "
         "by registry - CreditCardRecognizer supported languages: es, "
         "registry supported languages: en")
GENUINE = "spaCy model en_core_web_lg is older than the installed spaCy"


def _fake_presidio(monkeypatch, *, build_seconds: float):
    """Install a fake ``presidio_analyzer`` whose constructor takes a while
    and logs, at the END (after the slow part), the known noise plus one
    genuine warning - the ordering that made the race visible live."""
    counter = SimpleNamespace(constructions=0)
    lock = threading.Lock()

    class AnalyzerEngine:
        def __init__(self):
            with lock:
                counter.constructions += 1
            time.sleep(build_seconds)
            log = logging.getLogger(PRESIDIO_LOGGER)
            log.warning(NOISE)
            log.warning(GENUINE)

    module = ModuleType("presidio_analyzer")
    module.AnalyzerEngine = AnalyzerEngine
    monkeypatch.setitem(sys.modules, "presidio_analyzer", module)
    monkeypatch.setattr(pii, "_presidio_available", None)
    monkeypatch.setattr(pii, "_presidio_analyzer", None)
    return counter


@pytest.fixture
def presidio_logger_level():
    log = logging.getLogger(PRESIDIO_LOGGER)
    before = log.level
    yield log
    log.setLevel(before)


def test_concurrent_loaders_build_once_and_share_the_analyzer(monkeypatch, caplog, presidio_logger_level):
    counter = _fake_presidio(monkeypatch, build_seconds=0.3)
    with caplog.at_level(logging.WARNING, logger=PRESIDIO_LOGGER):
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(pii._get_presidio_analyzer)
            time.sleep(0.1)  # the second caller arrives mid-construction
            second = pool.submit(pii._get_presidio_analyzer)
            analyzers = [first.result(), second.result()]

    assert counter.constructions == 1, "spaCy must not be loaded twice"
    assert analyzers[0] is analyzers[1] is not None
    messages = [r.getMessage() for r in caplog.records if r.name == PRESIDIO_LOGGER]
    assert not any("not supported by registry" in m for m in messages), messages
    assert messages.count(GENUINE) == 1
    assert presidio_logger_level.level == logging.NOTSET, "the load must not leave the logger muted"
    assert pii.is_presidio_available()


def test_known_noise_is_dropped_but_other_presidio_warnings_survive(monkeypatch, caplog, presidio_logger_level):
    _fake_presidio(monkeypatch, build_seconds=0.0)
    with caplog.at_level(logging.WARNING, logger=PRESIDIO_LOGGER):
        assert pii._get_presidio_analyzer() is not None
    messages = [r.getMessage() for r in caplog.records if r.name == PRESIDIO_LOGGER]
    assert messages == [GENUINE]


def test_filter_is_removed_after_the_load(monkeypatch, caplog, presidio_logger_level):
    """The suppression is scoped to construction: the same text logged later
    by Presidio (or anyone) is not silently eaten for the life of the process."""
    _fake_presidio(monkeypatch, build_seconds=0.0)
    pii._get_presidio_analyzer()
    assert presidio_logger_level.filters == []
    with caplog.at_level(logging.WARNING, logger=PRESIDIO_LOGGER):
        presidio_logger_level.warning(NOISE)
    assert any(NOISE == r.getMessage() for r in caplog.records)


def test_failed_load_is_decided_once_and_releases_the_lock(monkeypatch, caplog):
    module = ModuleType("presidio_analyzer")

    class AnalyzerEngine:
        def __init__(self):
            raise OSError("[E050] Can't find model 'en_core_web_lg'")

    module.AnalyzerEngine = AnalyzerEngine
    monkeypatch.setitem(sys.modules, "presidio_analyzer", module)
    monkeypatch.setattr(pii, "_presidio_available", None)
    monkeypatch.setattr(pii, "_presidio_analyzer", None)
    assert pii._get_presidio_analyzer() is None
    assert pii._presidio_available is False
    # a second call must not deadlock or retry the construction
    assert pii._get_presidio_analyzer() is None
    assert not pii._load_lock.locked()
