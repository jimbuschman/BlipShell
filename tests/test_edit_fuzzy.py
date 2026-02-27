"""Tests for EditFileTool fuzzy matching layers."""

import os
import tempfile

import pytest

from blipshell.core.tools.filesystem import EditFileTool


@pytest.fixture
def temp_file():
    """Create a temporary file for edit tests."""
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.mark.asyncio
async def test_exact_match(temp_file):
    """Layer 1: exact string match works as before."""
    write_file(temp_file, "def hello():\n    print('hello')\n")
    tool = EditFileTool()

    result = await tool.execute(
        path=temp_file,
        old_text="print('hello')",
        new_text="print('world')",
    )

    assert "Successfully edited" in result
    assert "print('world')" in read_file(temp_file)
    # Exact match should NOT mention fuzzy or whitespace
    assert "fuzzy" not in result
    assert "whitespace" not in result


@pytest.mark.asyncio
async def test_whitespace_trailing_spaces(temp_file):
    """Layer 2: match when file has trailing spaces that old_text doesn't."""
    # File has trailing spaces after 'hello'
    write_file(temp_file, "def hello():   \n    print('hello')   \n")
    tool = EditFileTool()

    # old_text without trailing spaces
    result = await tool.execute(
        path=temp_file,
        old_text="def hello():\n    print('hello')",
        new_text="def hello():\n    print('world')",
    )

    assert "Successfully edited" in result
    assert "whitespace" in result.lower()
    assert "print('world')" in read_file(temp_file)


@pytest.mark.asyncio
async def test_whitespace_indentation_mismatch(temp_file):
    """Layer 2: match when indentation differs (tabs vs spaces, different depth)."""
    # File uses 4 spaces
    write_file(temp_file, "class Foo:\n    def bar(self):\n        return 42\n")
    tool = EditFileTool()

    # old_text uses 2 spaces (LLM hallucinated different indentation)
    result = await tool.execute(
        path=temp_file,
        old_text="class Foo:\n  def bar(self):\n    return 42",
        new_text="class Foo:\n  def bar(self):\n    return 99",
    )

    assert "Successfully edited" in result
    content = read_file(temp_file)
    # The edit should have been applied with the FILE's indentation
    assert "return 99" in content


@pytest.mark.asyncio
async def test_fuzzy_match_minor_difference(temp_file):
    """Layer 3: fuzzy match when there's a small typo/difference."""
    write_file(temp_file, "def calculate_total(items):\n    total = sum(item.price for item in items)\n    return total\n")
    tool = EditFileTool()

    # old_text has a minor difference ("item.cost" vs "item.price")
    result = await tool.execute(
        path=temp_file,
        old_text="def calculate_total(items):\n    total = sum(item.cost for item in items)\n    return total",
        new_text="def calculate_total(items, tax=0):\n    total = sum(item.price for item in items)\n    return total * (1 + tax)",
    )

    assert "Successfully edited" in result
    assert "fuzzy" in result.lower()
    content = read_file(temp_file)
    assert "tax" in content


@pytest.mark.asyncio
async def test_no_match_with_hint(temp_file):
    """When all layers fail, error includes closest match hint."""
    write_file(temp_file, "def alpha():\n    pass\n\ndef beta():\n    pass\n\ndef gamma():\n    pass\n")
    tool = EditFileTool()

    # Completely different text
    result = await tool.execute(
        path=temp_file,
        old_text="def totally_different_function():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z",
        new_text="def replacement():\n    pass",
    )

    assert "Error:" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_file_not_found(temp_file):
    """Non-existent file returns proper error."""
    tool = EditFileTool()
    result = await tool.execute(
        path="/nonexistent/path.py",
        old_text="anything",
        new_text="anything",
    )
    assert "Error:" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_exact_match_preferred_over_fuzzy(temp_file):
    """Exact match is used even when fuzzy would also match."""
    content = "x = 1\nx = 1\n"
    write_file(temp_file, content)
    tool = EditFileTool()

    result = await tool.execute(
        path=temp_file,
        old_text="x = 1",
        new_text="x = 2",
    )

    assert "Successfully edited" in result
    assert "fuzzy" not in result
    # Only first occurrence should be replaced
    file_content = read_file(temp_file)
    assert file_content.count("x = 2") == 1
    assert file_content.count("x = 1") == 1


@pytest.mark.asyncio
async def test_root_path_resolution(temp_file):
    """Root path is used for relative path resolution."""
    root = os.path.dirname(temp_file)
    basename = os.path.basename(temp_file)
    write_file(temp_file, "x = 'old'")
    tool = EditFileTool(root_path=root)

    result = await tool.execute(
        path=basename,
        old_text="x = 'old'",
        new_text="x = 'new'",
    )

    assert "Successfully edited" in result
    assert "x = 'new'" in read_file(temp_file)
