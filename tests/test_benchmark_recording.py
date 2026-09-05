"""Benchmark rethink (2026-08-10): transcripts persisted, repeats merged.

The old harness scored outputs and discarded them; single runs read as
precise points. These pin the three additions: every candidate call is
recorded at the router chokepoint, N repeats merge to mean+spread, and
transcripts land in a committed file next to the result rows.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from blipshell.benchmark.recording import (
    LIVE_CORPUS_MARKER, MAX_RESPONSE_CHARS, RecordingClient, RecordingRouter,
)
from blipshell.benchmark.results import ResultsStore
from blipshell.benchmark.runner import merge_repeat_rows


class TestRecordingRouter:
    def _wrapped(self, response="the answer"):
        inner = MagicMock()
        inner.generate = AsyncMock(return_value=response)
        return RecordingRouter(inner), inner

    async def test_records_every_call_with_labels(self):
        r, _ = self._wrapped()
        r.suite = "pipeline"
        r.repeat = 2

        out = await r.generate("summarization", "summarize this", system="be brief")

        assert out == "the answer"
        call = r.calls[0]
        assert call["suite"] == "pipeline"
        assert call["repeat"] == 2
        assert call["task_type"] == "summarization"
        assert call["prompt"] == "summarize this"
        assert call["system"] == "be brief"
        assert call["response"] == "the answer"
        assert call["error"] is None
        assert call["elapsed_s"] >= 0

    async def test_failures_are_recorded_and_reraised(self):
        """A timeout or refusal is exactly the transcript you want to read."""
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=RuntimeError("model wedged"))
        r = RecordingRouter(inner)

        with pytest.raises(RuntimeError):
            await r.generate("reasoning", "think hard")

        call = r.calls[0]
        assert "model wedged" in call["error"]
        assert call["response"] is None

    async def test_long_output_truncates_with_total(self):
        r, _ = self._wrapped(response="x" * 20000)
        await r.generate("reasoning", "p")
        resp = r.calls[0]["response"]
        assert len(resp) < 20000
        assert "20000 chars total" in resp

    async def test_delegates_everything_else(self):
        inner = MagicMock()
        inner.get_fallback_model.return_value = "fb"
        r = RecordingRouter(inner)
        assert r.get_fallback_model("coding") == "fb"


class TestMergeRepeats:
    def _row(self, metric, value, suite="pipeline", task="ranking"):
        return {"suite": suite, "task_type": task, "metric": metric,
                "value": value, "unit": "ratio", "raw": None}

    def test_numeric_metrics_get_mean_and_spread(self):
        merged = merge_repeat_rows([
            [self._row("accuracy", 0.420)],
            [self._row("accuracy", 0.345)],
        ])
        assert merged[0]["value"] == pytest.approx(0.3825)
        assert merged[0]["values"] == [0.420, 0.345]
        assert merged[0]["spread"] == pytest.approx(0.075)

    def test_single_run_is_untouched(self):
        rows = [self._row("accuracy", 0.9)]
        merged = merge_repeat_rows([rows])
        assert merged == rows
        assert "values" not in merged[0]

    def test_row_missing_from_one_repeat_merges_what_exists(self):
        """A judge that failed on one repeat must not sink the metric."""
        merged = merge_repeat_rows([
            [self._row("accuracy", 0.8), self._row("quality", 0.6)],
            [self._row("accuracy", 0.9)],
        ])
        by_metric = {m["metric"]: m for m in merged}
        assert by_metric["accuracy"]["value"] == pytest.approx(0.85)
        assert by_metric["quality"]["value"] == 0.6
        assert "values" not in by_metric["quality"]

    def test_non_numeric_values_keep_first(self):
        merged = merge_repeat_rows([
            [self._row("latency_note", None)],
            [self._row("latency_note", None)],
        ])
        assert merged[0]["value"] is None


class TestTranscriptFile:
    def test_written_next_to_results_and_readable(self, tmp_path):
        store = ResultsStore(tmp_path)
        calls = [{"suite": "pipeline", "repeat": 0, "task_type": "ranking",
                  "system": None, "prompt": "p", "response": "r",
                  "error": None, "elapsed_s": 1.0, "est_tok_s": 4.0}]

        path = store.write_transcripts(
            model="gpt-oss:latest", run_ts="2026-08-10T20:00:00+00:00",
            calls=calls,
        )

        assert path.name.endswith("__transcripts.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["model"] == "gpt-oss:latest"
        assert data["calls"][0]["prompt"] == "p"

    def test_spread_fields_survive_write_run(self, tmp_path):
        """_ROW_FIELDS filters unknown keys — values/spread must be known."""
        store = ResultsStore(tmp_path)
        path = store.write_run(
            model="m", run_group="g", run_ts="2026-08-10T20:00:00+00:00",
            rows=[{"suite": "s", "task_type": "t", "metric": "accuracy",
                   "value": 0.5, "unit": "ratio", "raw": None,
                   "values": [0.4, 0.6], "spread": 0.2}],
        )
        row = json.loads(path.read_text(encoding="utf-8"))["rows"][0]
        assert row["values"] == [0.4, 0.6]
        assert row["spread"] == 0.2


class TestRecordingClient:
    async def test_chat_calls_are_recorded_with_tool_names(self):
        """The pilot's gap: the 0.19 coding suite produced zero transcripts
        because client.chat bypasses router.generate()."""
        from blipshell.benchmark.recording import RecordingClient, RecordingRouter

        recorder = RecordingRouter(MagicMock())
        recorder.suite = "coding"
        inner = MagicMock()
        inner.chat = AsyncMock(return_value={"message": {
            "content": "on it", "tool_calls": [
                {"function": {"name": "read_file", "arguments": "{}"}}],
        }})
        client = RecordingClient(inner, recorder)

        out = await client.chat(messages=[{"role": "user", "content": "fix the bug"}],
                                model="m", tools=[])

        assert out["message"]["content"] == "on it"
        call = recorder.calls[0]
        assert call["suite"] == "coding"
        assert call["task_type"] == "chat"
        assert call["prompt"] == "fix the bug"
        assert call["tool_calls"] == ["read_file"]

    async def test_wrap_router_clients_wraps_endpoints(self):
        from blipshell.benchmark.harness import build_candidate_router
        from blipshell.benchmark.recording import (
            RecordingClient, RecordingRouter, wrap_router_clients,
        )

        router = RecordingRouter(build_candidate_router("m"))
        wrap_router_clients(router)

        eps = router._endpoint_manager._endpoints
        assert eps and all(isinstance(ep.client, RecordingClient) for ep in eps)


class TestLiveCorpusRedaction:
    """Suites that sample the real database must not write its text into the
    committed transcripts file (2026-09-02: 732 real messages reached GitHub).
    """

    def _wrapped(self, response="a summary of the message"):
        inner = MagicMock()
        inner.generate = AsyncMock(return_value=response)
        return RecordingRouter(inner), inner

    async def test_live_corpus_calls_keep_shape_but_not_text(self):
        r, _ = self._wrapped()
        r.suite = "realdata"
        r.live_corpus = True

        out = await r.generate("ranking", "Rate this message:\n\nmy real message",
                               system="be terse")

        assert out == "a summary of the message"  # caller still gets the answer
        call = r.calls[0]
        assert call["suite"] == "realdata"
        assert call["prompt"] == LIVE_CORPUS_MARKER
        assert call["system"] == LIVE_CORPUS_MARKER
        assert call["response"] == LIVE_CORPUS_MARKER
        assert "my real message" not in json.dumps(call)
        assert call["elapsed_s"] >= 0

    async def test_flag_off_records_text_as_before(self):
        r, _ = self._wrapped()
        r.suite = "pipeline"
        await r.generate("ranking", "synthetic case")
        assert r.calls[0]["prompt"] == "synthetic case"

    async def test_live_corpus_failure_keeps_error_without_text(self):
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=RuntimeError("wedged"))
        r = RecordingRouter(inner)
        r.live_corpus = True
        with pytest.raises(RuntimeError):
            await r.generate("ranking", "real text")
        call = r.calls[0]
        assert "wedged" in call["error"]
        assert call["prompt"] == LIVE_CORPUS_MARKER
        assert call["response"] is None

    async def test_client_layer_honours_the_flag(self):
        inner = MagicMock()
        inner.chat = AsyncMock(return_value={"message": {"content": "reply text"}})
        rec = RecordingRouter(MagicMock())
        rec.live_corpus = True
        client = RecordingClient(inner, rec)

        await client.chat(messages=[{"role": "user", "content": "real user text"}])

        call = rec.calls[0]
        assert call["prompt"] == LIVE_CORPUS_MARKER
        assert call["response"] == LIVE_CORPUS_MARKER
        assert "real user text" not in json.dumps(call)

    async def test_account_name_is_scrubbed_from_quota_errors(self):
        """Ollama cloud's 429 text names the account; keep the error, drop the name."""
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=RuntimeError(
            "ResponseError: you (somebody) have reached your session usage limit"))
        r = RecordingRouter(inner)
        with pytest.raises(RuntimeError):
            await r.generate("ranking", "p")
        err = r.calls[0]["error"]
        assert "somebody" not in err
        assert "you ([account]) have reached your session usage limit" in err

    def test_harness_label_sets_the_flag_for_live_suites(self):
        from blipshell.benchmark.harness import LIVE_CORPUS_SUITES
        assert "realdata" in LIVE_CORPUS_SUITES
        assert "pipeline" not in LIVE_CORPUS_SUITES
