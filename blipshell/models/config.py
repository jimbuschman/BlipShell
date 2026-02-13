"""Configuration Pydantic models matching config.yaml schema."""

from typing import Optional

from pydantic import BaseModel, Field


class ModelsConfig(BaseModel):
    """Model assignments for different task types."""
    reasoning: str = "qwen3:14b"
    tool_calling: str = "qwen3:14b"
    coding: str = "qwen3:14b"
    summarization: str = "gemma3:4b"
    ranking: str = "gemma3:4b"
    embedding: str = "nomic-embed-text"


class EndpointConfig(BaseModel):
    """Configuration for an Ollama endpoint."""
    name: str
    url: str = "http://localhost:11434"
    roles: list[str] = Field(default_factory=lambda: ["reasoning"])
    priority: int = 1
    max_concurrent: int = 2
    enabled: bool = True


class PoolConfig(BaseModel):
    """Configuration for a memory token budget pool."""
    percentage: float
    max_tokens: Optional[int] = None
    priority: int = 0


class MemoryPoolsConfig(BaseModel):
    """All memory pool configurations."""
    core: PoolConfig = PoolConfig(percentage=0.10, max_tokens=2048, priority=5)
    active_session: PoolConfig = PoolConfig(percentage=0.35, priority=3)
    recent_history: PoolConfig = PoolConfig(percentage=0.15, priority=4)
    recall: PoolConfig = PoolConfig(percentage=0.30, max_tokens=8192, priority=2)
    buffer: PoolConfig = PoolConfig(percentage=0.10, priority=1)


class MemoryConfig(BaseModel):
    """Memory system configuration."""
    pools: MemoryPoolsConfig = MemoryPoolsConfig()
    total_context_tokens: int = 32768
    system_prompt_reserve: int = 2048
    overflow_batch_size: int = 4
    recall_search_limit: int = 20
    min_rank_threshold: int = 3
    importance_recency_bonus: float = 0.1
    importance_tag_bonus: float = 0.05


class SessionConfig(BaseModel):
    """Session management configuration."""
    max_messages_before_summary: int = 50
    summary_chunk_size: int = 20
    auto_save_interval: int = 300


class AgentConfig(BaseModel):
    """Agent behavior configuration."""
    max_tool_iterations: int = 5
    system_prompt: str = (
        "You are BlipShell, a helpful local AI assistant with persistent memory.\n"
        "You remember previous conversations and learn from interactions.\n"
        "You have access to tools for file operations, shell commands, web search, "
        "and memory management.\n"
        "Be concise and helpful. Use your memory to provide personalized assistance."
    )
    stream: bool = True


class ShellToolConfig(BaseModel):
    """Shell tool configuration."""
    timeout: int = 30
    allowed_commands: list[str] = Field(default_factory=lambda: [
        "ls", "dir", "cat", "type", "echo", "pwd", "cd", "find", "grep",
        "head", "tail", "wc", "sort", "python", "pip", "git", "node", "npm",
        "cargo", "make", "cmake",
    ])


class FilesystemToolConfig(BaseModel):
    """Filesystem tool configuration."""
    max_file_size: int = 1048576
    blocked_paths: list[str] = Field(default_factory=lambda: ["/etc/shadow", "/etc/passwd"])


class WebToolConfig(BaseModel):
    """Web tool configuration."""
    max_fetch_size: int = 524288
    timeout: int = 15


class ToolsConfig(BaseModel):
    """All tool configurations."""
    shell: ShellToolConfig = ShellToolConfig()
    filesystem: FilesystemToolConfig = FilesystemToolConfig()
    web: WebToolConfig = WebToolConfig()


class NoiseConfig(BaseModel):
    """Noise filter configuration."""
    min_word_count: int = 3
    max_filler_ratio: float = 0.6


class TaggingConfig(BaseModel):
    """Tagging configuration."""
    max_tags: int = 7


class DatabaseConfig(BaseModel):
    """Database paths configuration."""
    path: str = "data/blipshell.db"
    chroma_path: str = "data/chroma"


class WebUIConfig(BaseModel):
    """Web UI configuration."""
    host: str = "0.0.0.0"
    port: int = 8000


class BlipShellConfig(BaseModel):
    """Root configuration model."""
    models: ModelsConfig = ModelsConfig()
    endpoints: list[EndpointConfig] = Field(default_factory=lambda: [
        EndpointConfig(
            name="local",
            url="http://localhost:11434",
            roles=["reasoning", "tool_calling", "coding", "embedding"],
            priority=1,
            max_concurrent=2,
        )
    ])
    memory: MemoryConfig = MemoryConfig()
    session: SessionConfig = SessionConfig()
    agent: AgentConfig = AgentConfig()
    tools: ToolsConfig = ToolsConfig()
    noise: NoiseConfig = NoiseConfig()
    tagging: TaggingConfig = TaggingConfig()
    database: DatabaseConfig = DatabaseConfig()
    web_ui: WebUIConfig = WebUIConfig()
