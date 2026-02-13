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
    context_tokens: Optional[int] = None  # per-endpoint context window override


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
    similarity_threshold: float = 0.5
    importance_boost_weight: float = 0.2
    search_overfetch_multiplier: int = 2
    auto_prune_days: int = 90
    prune_max_importance: float = 0.3
    prune_max_rank: int = 2


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
        "Be concise and helpful. Use your memory to provide personalized assistance.\n\n"
        "IMPORTANT: You have tools available. You MUST use them when appropriate:\n"
        "- Use web_search to look up current information or answer factual questions.\n"
        "- Use web_fetch to read the contents of a specific URL.\n"
        "- Use read_file, write_file, list_directory for file operations.\n"
        "- Use run_command to execute shell commands.\n"
        "- Use search_memories to recall past conversations.\n"
        "- Use save_core_memory to permanently remember important facts about the user.\n"
        "- Use promote_to_core_memory to promote an important memory or lesson to permanent core memory.\n"
        "When a user asks you to search or you need current information, ALWAYS call web_search."
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


class LLMConfig(BaseModel):
    """LLM call configuration."""
    max_retries: int = 2
    retry_base_delay: float = 1.0


class AuthConfig(BaseModel):
    """Web UI authentication configuration."""
    enabled: bool = False
    api_key: str = ""


class WebUIConfig(BaseModel):
    """Web UI configuration."""
    host: str = "0.0.0.0"
    port: int = 8000


class PlannerConfig(BaseModel):
    """Task planner configuration."""
    enabled: bool = True
    auto_approve: bool = True
    max_steps: int = 7
    max_retries_per_step: int = 2
    complexity_threshold_words: int = 20


class WorkerConfig(BaseModel):
    """Remote worker configuration."""
    enabled: bool = False
    default_remote_endpoint: str = ""
    task_types_for_remote: list[str] = Field(default_factory=lambda: [
        "summarization", "research", "analysis",
    ])
    poll_interval: int = 10


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
    llm: LLMConfig = LLMConfig()
    auth: AuthConfig = AuthConfig()
    database: DatabaseConfig = DatabaseConfig()
    web_ui: WebUIConfig = WebUIConfig()
    planner: PlannerConfig = PlannerConfig()
    worker: WorkerConfig = WorkerConfig()
