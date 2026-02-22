"""Tests for AST-based repo map."""

import os
import tempfile

import pytest

from blipshell.core.repo_map import RepoMap, _extract_defs


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project with Python files."""
    # Main module
    pkg = tmp_path / "myapp"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")

    (pkg / "models.py").write_text(
        "class User:\n"
        "    def __init__(self, name, email):\n"
        "        self.name = name\n"
        "        self.email = email\n"
        "\n"
        "    def full_name(self):\n"
        "        return self.name\n"
        "\n"
        "    def __repr__(self):\n"
        "        return f'User({self.name})'\n"
        "\n"
        "class Post:\n"
        "    def __init__(self, title, body, author):\n"
        "        self.title = title\n"
        "\n"
        "    def publish(self):\n"
        "        pass\n"
    )

    (pkg / "utils.py").write_text(
        "def calculate_hash(data):\n"
        "    pass\n"
        "\n"
        "async def fetch_data(url, timeout=30):\n"
        "    pass\n"
        "\n"
        "def _private_helper():\n"
        "    pass\n"
    )

    # Config file at root (not a package)
    (tmp_path / "setup.py").write_text(
        "from setuptools import setup\n"
        "setup(name='myapp')\n"
    )

    # A file with syntax errors
    (pkg / "broken.py").write_text("def oops(\n")

    return tmp_path


def test_build_basic(temp_project):
    """Build produces a non-empty map for a valid project."""
    repo_map = RepoMap(str(temp_project))
    result = repo_map.build()

    assert result  # non-empty
    assert "class User" in result
    assert "class Post" in result
    assert "calculate_hash" in result
    assert "async fetch_data" in result


def test_class_methods_shown(temp_project):
    """Class methods (except dunders besides __init__) are listed."""
    repo_map = RepoMap(str(temp_project))
    result = repo_map.build()

    assert "__init__" in result
    assert "full_name" in result
    assert "publish" in result
    # __repr__ should be filtered out
    assert "__repr__" not in result


def test_function_args_shown(temp_project):
    """Function arguments are shown in compact form."""
    repo_map = RepoMap(str(temp_project))
    result = repo_map.build()

    assert "name, email" in result
    assert "url, timeout" in result
    assert "data" in result


def test_caching(temp_project):
    """Second build uses cache (same mtime = no re-parse)."""
    repo_map = RepoMap(str(temp_project))
    result1 = repo_map.build()
    result2 = repo_map.build()

    assert result1 == result2
    # Cache should have entries
    assert len(repo_map._cache) > 0


def test_cache_invalidation(temp_project):
    """Invalidated files get re-parsed."""
    repo_map = RepoMap(str(temp_project))
    repo_map.build()

    # Modify a file
    models = temp_project / "myapp" / "models.py"
    models.write_text(
        "class NewModel:\n"
        "    def new_method(self):\n"
        "        pass\n"
    )

    # Invalidate and rebuild
    repo_map.invalidate("myapp/models.py")
    result = repo_map.build()

    assert "class NewModel" in result
    assert "class User" not in result


def test_broken_file_handled(temp_project):
    """Files with syntax errors are silently skipped."""
    repo_map = RepoMap(str(temp_project))
    result = repo_map.build()

    # Should still produce output from valid files
    assert "class User" in result
    # Broken file should not crash
    assert "broken.py" not in result or "class" not in result.split("broken.py")[-1].split("\n")[0]


def test_skip_dirs(temp_project):
    """__pycache__ and other skip dirs are excluded."""
    pycache = temp_project / "myapp" / "__pycache__"
    pycache.mkdir()
    (pycache / "models.cpython-311.pyc").write_text("")

    repo_map = RepoMap(str(temp_project))
    result = repo_map.build()

    assert "__pycache__" not in result


def test_empty_project(tmp_path):
    """Empty project produces empty map."""
    repo_map = RepoMap(str(tmp_path))
    result = repo_map.build()
    assert result == ""


def test_relative_paths(temp_project):
    """Paths in output are relative to project root."""
    repo_map = RepoMap(str(temp_project))
    result = repo_map.build()

    # Should use forward slashes and be relative
    assert "myapp/models.py" in result
    assert str(temp_project) not in result


def test_max_lines_limit(tmp_path):
    """Output respects max_lines parameter."""
    # Create many files
    pkg = tmp_path / "big"
    pkg.mkdir()
    for i in range(50):
        (pkg / f"module_{i}.py").write_text(
            f"class Class{i}:\n"
            f"    def method_{i}(self):\n"
            f"        pass\n"
        )

    repo_map = RepoMap(str(tmp_path))
    result = repo_map.build(max_lines=20)

    assert len(result.splitlines()) <= 22  # small buffer for truncation message
