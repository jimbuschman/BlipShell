"""Tests for multi-language repo map."""

import os
import tempfile

import pytest

from blipshell.core.repo_map import (
    RepoMap, _extract_defs, _extract_js_ts, _extract_go,
    _extract_rust, _extract_java, _extract_c_cpp,
)
from blipshell.core.tools.code_tools import RepoMapTool


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


# ---------------------------------------------------------------------------
# JavaScript / TypeScript extraction
# ---------------------------------------------------------------------------

@pytest.fixture
def js_project(tmp_path):
    """Create a temporary project with JS/TS files."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "app.ts").write_text(
        "export class Router {\n"
        "  private routes: Map<string, Handler>;\n"
        "\n"
        "  constructor() {\n"
        "    this.routes = new Map();\n"
        "  }\n"
        "\n"
        "  addRoute(path: string, handler: Handler) {\n"
        "    this.routes.set(path, handler);\n"
        "  }\n"
        "\n"
        "  async handleRequest(req: Request) {\n"
        "    return null;\n"
        "  }\n"
        "}\n"
        "\n"
        "export interface Handler {\n"
        "  handle(req: Request): Promise<Response>;\n"
        "}\n"
        "\n"
        "export type RouteConfig = {\n"
        "  path: string;\n"
        "  method: string;\n"
        "};\n"
        "\n"
        "export enum HttpMethod {\n"
        "  GET = 'GET',\n"
        "  POST = 'POST',\n"
        "}\n"
    )

    (src / "utils.js").write_text(
        "export function formatDate(date) {\n"
        "  return date.toISOString();\n"
        "}\n"
        "\n"
        "export async function fetchData(url, options) {\n"
        "  return fetch(url, options);\n"
        "}\n"
        "\n"
        "const debounce = (fn, delay) => {\n"
        "  let timer;\n"
        "  return (...args) => {\n"
        "    clearTimeout(timer);\n"
        "    timer = setTimeout(() => fn(...args), delay);\n"
        "  };\n"
        "};\n"
    )

    return tmp_path


def test_js_ts_class_extraction(js_project):
    """TypeScript class with methods is extracted."""
    repo_map = RepoMap(str(js_project))
    result = repo_map.build()

    assert "class Router" in result
    assert "addRoute" in result
    assert "handleRequest" in result


def test_js_ts_interface_enum_type(js_project):
    """TS interfaces, enums, and type aliases are extracted."""
    repo_map = RepoMap(str(js_project))
    result = repo_map.build()

    assert "interface Handler" in result
    assert "type RouteConfig" in result
    assert "enum HttpMethod" in result


def test_js_function_extraction(js_project):
    """JS functions (export, async, arrow const) are extracted."""
    repo_map = RepoMap(str(js_project))
    result = repo_map.build()

    assert "formatDate" in result
    assert "fetchData" in result
    assert "debounce" in result


# ---------------------------------------------------------------------------
# Go extraction
# ---------------------------------------------------------------------------

@pytest.fixture
def go_project(tmp_path):
    """Create a temporary Go project."""
    (tmp_path / "main.go").write_text(
        "package main\n"
        "\n"
        "import \"fmt\"\n"
        "\n"
        "type Server struct {\n"
        "    host string\n"
        "    port int\n"
        "}\n"
        "\n"
        "type Handler interface {\n"
        "    ServeHTTP(w ResponseWriter, r *Request)\n"
        "}\n"
        "\n"
        "func NewServer(host string, port int) *Server {\n"
        "    return &Server{host: host, port: port}\n"
        "}\n"
        "\n"
        "func (s *Server) Start() error {\n"
        "    return nil\n"
        "}\n"
        "\n"
        "func main() {\n"
        "    fmt.Println(\"hello\")\n"
        "}\n"
    )
    return tmp_path


def test_go_struct_interface(go_project):
    """Go structs and interfaces are extracted."""
    repo_map = RepoMap(str(go_project))
    result = repo_map.build()

    assert "struct Server" in result
    assert "interface Handler" in result


def test_go_functions(go_project):
    """Go functions and methods are extracted."""
    repo_map = RepoMap(str(go_project))
    result = repo_map.build()

    assert "NewServer" in result
    assert "Start" in result
    assert "main()" in result


# ---------------------------------------------------------------------------
# Rust extraction
# ---------------------------------------------------------------------------

@pytest.fixture
def rust_project(tmp_path):
    """Create a temporary Rust project."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "lib.rs").write_text(
        "pub struct Config {\n"
        "    pub host: String,\n"
        "    pub port: u16,\n"
        "}\n"
        "\n"
        "pub enum Status {\n"
        "    Active,\n"
        "    Inactive,\n"
        "}\n"
        "\n"
        "pub trait Service {\n"
        "    fn start(&self) -> Result<(), Error>;\n"
        "}\n"
        "\n"
        "impl Config {\n"
        "    pub fn new(host: String, port: u16) -> Self {\n"
        "        Config { host, port }\n"
        "    }\n"
        "}\n"
        "\n"
        "impl Service for Config {\n"
        "    fn start(&self) -> Result<(), Error> {\n"
        "        Ok(())\n"
        "    }\n"
        "}\n"
        "\n"
        "pub async fn run_server(config: &Config) -> Result<(), Error> {\n"
        "    Ok(())\n"
        "}\n"
    )
    return tmp_path


def test_rust_types(rust_project):
    """Rust structs, enums, traits, and impls are extracted."""
    repo_map = RepoMap(str(rust_project))
    result = repo_map.build()

    assert "struct Config" in result
    assert "enum Status" in result
    assert "trait Service" in result
    assert "impl Config" in result
    assert "impl Service for Config" in result


def test_rust_functions(rust_project):
    """Rust top-level functions are extracted."""
    repo_map = RepoMap(str(rust_project))
    result = repo_map.build()

    assert "run_server" in result
    # Note: impl methods (like `new`) require brace-tracking to extract.
    # Top-level fn and type definitions are the primary value.


# ---------------------------------------------------------------------------
# Java extraction
# ---------------------------------------------------------------------------

@pytest.fixture
def java_project(tmp_path):
    """Create a temporary Java project."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "Server.java").write_text(
        "package com.example;\n"
        "\n"
        "public class Server {\n"
        "    private String host;\n"
        "    private int port;\n"
        "\n"
        "    public Server(String host, int port) {\n"
        "        this.host = host;\n"
        "        this.port = port;\n"
        "    }\n"
        "\n"
        "    public void start() {\n"
        "        // start server\n"
        "    }\n"
        "\n"
        "    public String getHost() {\n"
        "        return host;\n"
        "    }\n"
        "\n"
        "    private void internalMethod() {\n"
        "        // private\n"
        "    }\n"
        "}\n"
    )

    (src / "Handler.java").write_text(
        "package com.example;\n"
        "\n"
        "public interface Handler {\n"
        "    void handle(Request req);\n"
        "}\n"
    )

    return tmp_path


def test_java_class_methods(java_project):
    """Java classes with methods are extracted."""
    repo_map = RepoMap(str(java_project))
    result = repo_map.build()

    assert "class Server" in result
    assert "start" in result
    assert "getHost" in result


def test_java_interface(java_project):
    """Java interfaces are extracted."""
    repo_map = RepoMap(str(java_project))
    result = repo_map.build()

    assert "Handler" in result


# ---------------------------------------------------------------------------
# C/C++ extraction
# ---------------------------------------------------------------------------

@pytest.fixture
def c_project(tmp_path):
    """Create a temporary C project."""
    (tmp_path / "server.c").write_text(
        "#include <stdio.h>\n"
        "\n"
        "struct Config {\n"
        "    char *host;\n"
        "    int port;\n"
        "};\n"
        "\n"
        "int server_start(struct Config *config) {\n"
        "    return 0;\n"
        "}\n"
        "\n"
        "void server_stop(struct Config *config) {\n"
        "    // cleanup\n"
        "}\n"
    )

    (tmp_path / "utils.h").write_text(
        "#ifndef UTILS_H\n"
        "#define UTILS_H\n"
        "\n"
        "typedef struct {\n"
        "    int x;\n"
        "    int y;\n"
        "} Point;\n"
        "\n"
        "enum Color {\n"
        "    RED,\n"
        "    GREEN,\n"
        "    BLUE\n"
        "};\n"
        "\n"
        "#endif\n"
    )

    return tmp_path


def test_c_struct_enum(c_project):
    """C structs and enums are extracted."""
    repo_map = RepoMap(str(c_project))
    result = repo_map.build()

    assert "struct Config" in result
    assert "enum Color" in result


def test_c_functions(c_project):
    """C functions are extracted."""
    repo_map = RepoMap(str(c_project))
    result = repo_map.build()

    assert "server_start" in result
    assert "server_stop" in result


# ---------------------------------------------------------------------------
# Filtering tests
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_lang_project(tmp_path):
    """Create a project with multiple languages."""
    py_dir = tmp_path / "backend"
    py_dir.mkdir()
    (py_dir / "app.py").write_text(
        "class App:\n"
        "    def run(self):\n"
        "        pass\n"
    )

    ts_dir = tmp_path / "frontend"
    ts_dir.mkdir()
    (ts_dir / "index.ts").write_text(
        "export class Component {\n"
        "  render() {\n"
        "    return null;\n"
        "  }\n"
        "}\n"
    )

    (tmp_path / "main.go").write_text(
        "package main\n"
        "\n"
        "func main() {\n"
        "}\n"
    )

    return tmp_path


def test_language_filter(multi_lang_project):
    """Language filter restricts to one language."""
    repo_map = RepoMap(str(multi_lang_project))

    py_result = repo_map.build(language_filter="python")
    assert "class App" in py_result
    assert "Component" not in py_result
    assert "main()" not in py_result

    ts_result = repo_map.build(language_filter="typescript")
    assert "class Component" in ts_result
    assert "class App" not in ts_result


def test_path_filter(multi_lang_project):
    """Path filter restricts to subdirectory."""
    repo_map = RepoMap(str(multi_lang_project))

    backend = repo_map.build(path_filter="backend")
    assert "class App" in backend
    assert "Component" not in backend

    frontend = repo_map.build(path_filter="frontend")
    assert "class Component" in frontend
    assert "class App" not in frontend


def test_symbol_query(multi_lang_project):
    """Symbol query filters to matching definitions."""
    repo_map = RepoMap(str(multi_lang_project))

    result = repo_map.build(symbol_query="App")
    assert "class App" in result
    assert "Component" not in result
    assert "main" not in result


def test_symbol_query_no_match(multi_lang_project):
    """Symbol query with no matches returns message."""
    repo_map = RepoMap(str(multi_lang_project))
    result = repo_map.build(symbol_query="NonExistentSymbol")
    assert "No symbols matching" in result


def test_combined_filters(multi_lang_project):
    """Path + language + query filters combine correctly."""
    repo_map = RepoMap(str(multi_lang_project))

    result = repo_map.build(path_filter="backend", language_filter="python", symbol_query="run")
    assert "run" in result
    assert "Component" not in result


# ---------------------------------------------------------------------------
# RepoMapTool tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repo_map_tool_basic(temp_project):
    """RepoMapTool returns code map."""
    repo_map = RepoMap(str(temp_project))
    tool = RepoMapTool(repo_map)

    result = await tool.execute()
    assert "class User" in result
    assert "calculate_hash" in result


@pytest.mark.asyncio
async def test_repo_map_tool_with_query(temp_project):
    """RepoMapTool filters by symbol query."""
    repo_map = RepoMap(str(temp_project))
    tool = RepoMapTool(repo_map)

    result = await tool.execute(query="User")
    assert "class User" in result
    assert "calculate_hash" not in result


@pytest.mark.asyncio
async def test_repo_map_tool_no_results(tmp_path):
    """RepoMapTool returns helpful message on empty results."""
    repo_map = RepoMap(str(tmp_path))
    tool = RepoMapTool(repo_map)

    result = await tool.execute()
    assert "No code definitions found" in result


@pytest.mark.asyncio
async def test_repo_map_tool_definition():
    """RepoMapTool has correct definition."""
    repo_map = RepoMap(".")
    tool = RepoMapTool(repo_map)

    defn = tool.definition()
    assert defn.name == "repo_map"
    assert tool.read_only is True
    param_names = {p.name for p in defn.parameters}
    assert param_names == {"path", "language", "query", "max_lines"}
