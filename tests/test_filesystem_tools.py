"""Tests for filesystem tools. Focused on the read_file binary/image guard —
reading a PNG as text used to return replacement-char gibberish, which made the
model report it "couldn't see" attached images. Pure/deterministic (no Ollama).
"""

import pytest

from blipshell.core.tools.filesystem import ReadFileTool


@pytest.mark.asyncio
async def test_read_file_refuses_image_with_guidance(tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)  # PNG magic + nulls
    out = await ReadFileTool().execute(path=str(img))
    assert "gibberish" not in out
    assert "image" in out.lower()
    assert "cannot be read as text" in out
    assert "visual input" in out  # points the model at vision input


@pytest.mark.asyncio
async def test_read_file_refuses_binary_with_null_bytes(tmp_path):
    blob = tmp_path / "data.bin"
    blob.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 32)
    out = await ReadFileTool().execute(path=str(blob))
    assert "binary file" in out.lower()
    assert "null bytes" in out


@pytest.mark.asyncio
async def test_read_file_still_reads_text(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("line one\nline two\n", encoding="utf-8")
    out = await ReadFileTool().execute(path=str(f))
    assert "line one" in out
    assert "line two" in out
