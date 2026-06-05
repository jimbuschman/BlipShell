"""Tests for image/vision input — path detection, load/downscale/store, and the
per-provider message-shape translation. Pure/deterministic (no Ollama).

The behavioral end (actually sending an image to minimax-m3 and getting a
description, plus persist-&-replay and degradation) is validated on the Ollama PC.
"""

import base64
from io import BytesIO
from pathlib import Path

import pytest

from blipshell.core import vision
from blipshell.core.vision import (
    ImageRef,
    apply_images_ollama,
    apply_images_openai,
    extract_image_paths,
    has_image_refs,
    load_image,
    strip_image_refs,
)
from blipshell.llm.model_settings import ModelSettingsRegistry

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _make_png(path: Path, size=(64, 64), color=(255, 0, 0)):
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


# ── extract_image_paths ──────────────────────────────────────────────────────

def test_extract_finds_existing_image(tmp_path):
    img = _make_png(tmp_path / "shot.png")
    msg = f"what's wrong here? {img}"
    cleaned, paths = extract_image_paths(msg)
    assert paths == [str(img)]
    assert "shot.png" not in cleaned
    assert "what's wrong here?" in cleaned


def test_extract_ignores_nonexistent_path():
    cleaned, paths = extract_image_paths("look at C:/nope/missing.png please")
    assert paths == []
    assert cleaned == "look at C:/nope/missing.png please"


def test_extract_ignores_non_image_files(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    _cleaned, paths = extract_image_paths(f"read {f}")
    assert paths == []


def test_extract_no_paths_returns_message_unchanged():
    msg = "just a normal question about the code"
    assert extract_image_paths(msg) == (msg, [])


# ── load_image / encode ──────────────────────────────────────────────────────

def test_load_image_returns_ref_and_stores(tmp_path):
    src = _make_png(tmp_path / "a.png")
    store = tmp_path / "store"
    ref = load_image(str(src), store_dir=store)
    assert ref is not None
    assert ref.mime == "image/png"
    assert ref.orig_name == "a.png"
    assert Path(ref.path).is_file()
    assert Path(ref.path).parent == store


def test_load_image_downscales_large(tmp_path):
    src = _make_png(tmp_path / "big.png", size=(2000, 1500))
    ref = load_image(str(src), store_dir=tmp_path / "s")
    w, h = Image.open(ref.path).size
    assert max(w, h) <= vision.MAX_DIMENSION


def test_load_image_dedups_by_hash(tmp_path):
    src = _make_png(tmp_path / "dup.png")
    store = tmp_path / "s"
    r1 = load_image(str(src), store_dir=store)
    r2 = load_image(str(src), store_dir=store)
    assert r1.sha256 == r2.sha256
    assert r1.path == r2.path
    assert len(list(store.iterdir())) == 1


def test_load_image_missing_returns_none(tmp_path):
    assert load_image(str(tmp_path / "ghost.png"), store_dir=tmp_path) is None


def test_encode_for_send_roundtrips(tmp_path):
    src = _make_png(tmp_path / "e.png")
    ref = load_image(str(src), store_dir=tmp_path / "s")
    b64 = vision.encode_for_send(ref)
    raw = base64.b64decode(b64)
    assert Image.open(BytesIO(raw)).size  # decodes as a valid image


# ── message-shape translation ────────────────────────────────────────────────

def _msg_with_image(tmp_path):
    ref = load_image(str(_make_png(tmp_path / "m.png")), store_dir=tmp_path / "s")
    return {"role": "user", "content": "what is this?", "_image_refs": [ref.to_dict()]}


def test_has_image_refs(tmp_path):
    assert has_image_refs([_msg_with_image(tmp_path)]) is True
    assert has_image_refs([{"role": "user", "content": "hi"}]) is False


def test_apply_images_ollama(tmp_path):
    out = apply_images_ollama([_msg_with_image(tmp_path)])
    m = out[0]
    assert "_image_refs" not in m
    assert isinstance(m["images"], list) and len(m["images"]) == 1
    assert base64.b64decode(m["images"][0])  # valid base64
    assert m["content"] == "what is this?"


def test_apply_images_openai(tmp_path):
    out = apply_images_openai([_msg_with_image(tmp_path)])
    m = out[0]
    assert "_image_refs" not in m
    assert isinstance(m["content"], list)
    assert m["content"][0] == {"type": "text", "text": "what is this?"}
    assert m["content"][1]["type"] == "image_url"
    assert m["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_translation_noop_without_images():
    msgs = [{"role": "user", "content": "plain text"}]
    assert apply_images_ollama(msgs) is msgs
    assert apply_images_openai(msgs) is msgs


def test_strip_image_refs_degrades_to_text(tmp_path):
    out = strip_image_refs([_msg_with_image(tmp_path)])
    m = out[0]
    assert "_image_refs" not in m
    assert "images" not in m
    assert "m.png" in m["content"]
    assert "no vision endpoint" in m["content"]


# ── vision capability lookup ─────────────────────────────────────────────────

def test_is_vision_lookup():
    reg = ModelSettingsRegistry()
    reg.load({
        "minimax-m3": {"vision_capable": True},
        "minimax/minimax-m3": {"vision_capable": True},
        "gpt-oss": {"think": False},
    })
    assert reg.is_vision("minimax-m3:cloud") is True       # base-name match on ':'
    assert reg.is_vision("minimax/minimax-m3") is True      # exact match
    assert reg.is_vision("gpt-oss:latest") is False
    assert reg.is_vision("qwen3:14b") is False              # unknown → default False
