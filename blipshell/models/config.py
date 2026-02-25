"""Configuration Pydantic models matching config.yaml schema."""

import os
from typing import Optional

from pydantic import BaseModel, Field


def resolve_env_vars(value: Optional[str]) -> Optional[str]:
    """Expand ${ENV_VAR} syntax in a string to the environment variable value."""
    if value and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


class ModelsConfig(BaseModel):
    """Model assignments for different task types."""
    reasoning: str = "qwen3:14b"
    tool_calling: str = "qwen3:14b"
    coding: str = "qwen3:14b"
    summarization: str = "glm4:latest"
    ranking: str = "gemma3:4b"
    importance: str = "qwen3:14b"
    ranking_importance: Optional[str] = None  # combined scoring; falls back to ranking model
    embedding: str = "nomic-embed-text"
    # Fallback models used when cloud endpoints are unavailable
    reasoning_fallback: str = "gpt-oss:latest"
    tool_calling_fallback: str = "gpt-oss:latest"
    coding_fallback: str = "gpt-oss:latest"
    summarization_fallback: Optional[str] = None
    ranking_fallback: Optional[str] = None
    importance_fallback: Optional[str] = None
    ranking_importance_fallback: Optional[str] = None
    # Disable thinking for fallback models (faster conversational responses)
    fallback_think: bool = False


class EndpointConfig(BaseModel):
    """Configuration for an LLM endpoint (Ollama or OpenAI-compatible)."""
    name: str
    url: str = "http://localhost:11434"
    provider: str = "ollama"  # "ollama" or "openai" (OpenAI-compatible)
    api_key: Optional[str] = None  # API key, supports ${ENV_VAR} syntax
    roles: list[str] = Field(default_factory=lambda: ["reasoning"])
    priority: int = 1
    max_concurrent: int = 2
    enabled: bool = True
    context_tokens: Optional[int] = None  # per-endpoint context window override
    rate_limit_rpm: Optional[int] = None  # max requests per minute
    rate_limit_rpd: Optional[int] = None  # max requests per day
    models: dict[str, str] = Field(default_factory=dict)  # per-endpoint model overrides


class PoolConfig(BaseModel):
    """Configuration for a memory token budget pool."""
    percentage: float
    max_tokens: Optional[int] = None
    priority: int = 0


class MemoryPoolsConfig(BaseModel):
    """All memory pool configurations."""
    core: PoolConfig = PoolConfig(percentage=0.10, priority=5)
    active_session: PoolConfig = PoolConfig(percentage=0.35, priority=3)
    recent_history: PoolConfig = PoolConfig(percentage=0.15, priority=4)
    recall: PoolConfig = PoolConfig(percentage=0.30, priority=2)
    buffer: PoolConfig = PoolConfig(percentage=0.10, priority=1)


class DecayRatesConfig(BaseModel):
    """Per-memory-type temporal decay rates.

    Decay formula: score *= exp(-decay_rate * hours_age)
    Lower rate = slower decay = longer half-life.
    """
    fact: float = 0.0002       # ~50% after 144 days
    preference: float = 0.0003  # ~50% after 96 days
    skill: float = 0.0001      # ~50% after 289 days
    event: float = 0.002       # ~50% after 14 days
    conversation: float = 0.001  # ~50% after 29 days (current default)

    def get(self, memory_type: str) -> float:
        """Get decay rate for a memory type, falling back to conversation rate."""
        return getattr(self, memory_type, self.conversation)


class DedupConfig(BaseModel):
    """Memory deduplication configuration."""
    enabled: bool = True
    similarity_threshold: float = 0.7  # min cosine similarity to consider as duplicate candidate


class EntityResolutionConfig(BaseModel):
    """Entity resolution configuration."""
    enabled: bool = True  # backfill runs automatically on first startup
    embedding_auto_merge_threshold: float = 0.85  # auto-merge above this
    llm_arbitration_threshold: float = 0.70  # ask LLM between this and auto_merge
    max_candidates: int = 5  # max similar entities to check


class MemoryConfig(BaseModel):
    """Memory system configuration."""
    pools: MemoryPoolsConfig = MemoryPoolsConfig()
    total_context_tokens: int = 65536
    system_prompt_reserve: int = 2048
    overflow_batch_size: int = 4
    recall_search_limit: int = 20
    min_rank_threshold: int = 3
    importance_recency_bonus: float = 0.1
    importance_tag_bonus: float = 0.05
    similarity_threshold: float = 0.3  # lowered from 0.5 — nomic scores are lower but still relevant
    importance_boost_weight: float = 0.2
    tag_overlap_boost: float = 0.1
    search_overfetch_multiplier: int = 2
    decay_rate: float = 0.001  # temporal decay rate (~50% after 29 days) — global fallback
    decay_rates: DecayRatesConfig = DecayRatesConfig()
    fts_weight: float = 0.3  # weight for FTS5 RRF boost in hybrid search
    auto_prune_days: int = 0  # 0 = disabled; was 90 but archived 1083 imported memories
    prune_max_importance: float = 0.3
    prune_max_rank: int = 2
    consolidation_similarity: float = 0.85  # min cosine similarity to merge
    consolidation_batch_size: int = 0  # 0 = disabled; was 100 but deletes data silently
    contradiction_similarity_threshold: float = 0.7  # min similarity to check for contradiction
    tag_discovery_interval_days: int = 7  # days between discovery runs
    tag_discovery_sample_size: int = 20  # poorly-tagged memories to sample
    entity_extraction_batch_size: int = 50  # memories processed per startup run
    entity_boost: float = 0.15  # boost for memories found via entity graph
    project_boost: float = 0.15  # boost for memories from active project sessions
    score_floor_ratio: float = 0.6  # results must be within this ratio of top score
    min_score_floor: float = 0.4  # absolute minimum boosted_score to keep
    dedup_jaccard_threshold: float = 0.65  # Jaccard similarity to consider summaries duplicate
    project_session_limit: int = 50  # max recent project sessions for two-pass search
    dedup: DedupConfig = DedupConfig()
    entity_resolution: EntityResolutionConfig = EntityResolutionConfig()


class SessionConfig(BaseModel):
    """Session management configuration."""
    max_messages_before_summary: int = 50
    summary_chunk_size: int = 20
    auto_save_interval: int = 300


class AgentConfig(BaseModel):
    """Agent behavior configuration."""
    max_tool_iterations: int = 5
    system_prompt: str = (
        "You are BlipShell, a local AI assistant with persistent memory.\n"
        "You remember past conversations and use that context to give personalized, relevant answers.\n\n"
        "# How to Respond\n"
        "Be concise and direct. Answer questions conversationally.\n"
        "Only use tools when the task genuinely requires them — don't call tools just to look thorough.\n"
        "Do NOT narrate your process. Never say 'Let me search...' or 'I'll look that up...' "
        "— just do it or answer from what you know.\n\n"
        "# When to Use Tools\n"
        "- User asks a factual question you're unsure about -> web_search\n"
        "- User gives a URL to read -> web_fetch\n"
        "- User asks to read, create, or modify files -> read_file, write_file, edit_file\n"
        "- User asks to run a command -> run_command\n"
        "- You learn an important fact about the user -> save_core_memory\n"
        "- User asks to search the web -> web_search (always — never say you can't search)\n"
        "Do NOT use tools for questions you can answer from context or memory.\n\n"
        "# Memory\n"
        "Past conversations and lessons are automatically loaded into your context.\n"
        "Do NOT call search_memories unless the user explicitly asks to search for something specific.\n"
        "Use save_core_memory sparingly — only for important, lasting facts.\n\n"
        "# Working Style\n"
        "- Read a file before editing it. Do not propose changes to code you haven't read.\n"
        "- Make minimal changes. Do exactly what was asked — no refactoring or extra features.\n"
        "- If something fails twice, ask the user instead of retrying blindly.\n"
        "- Do NOT launch interactive or full-screen apps via run_command.\n"
    )
    stream: bool = True
    # Tool approval: tools listed here require user confirmation before execution
    tools_requiring_approval: list[str] = Field(default_factory=lambda: [
        "write_file", "edit_file", "run_command", "git_add", "git_commit",
    ])
    auto_approve_tools: bool = False  # bypass approval for all tools


class ShellToolConfig(BaseModel):
    """Shell tool configuration."""
    timeout: int = 30
    allowed_commands: list[str] = Field(default_factory=lambda: [
        "ls", "dir", "cat", "type", "echo", "pwd", "cd", "find", "grep",
        "head", "tail", "wc", "sort", "python", "pip", "git", "node", "npm",
        "cargo", "make", "cmake", "powershell", "pwsh",
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
    timeout: float = 120.0  # per-call timeout in seconds


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


def get_ollama_url(endpoints: list[EndpointConfig]) -> str:
    """Get the URL of the first Ollama endpoint (for ChromaDB embedding)."""
    for ep in endpoints:
        if ep.provider == "ollama" and ep.enabled:
            return ep.url
    return "http://localhost:11434"


class BlipShellConfig(BaseModel):
    """Root configuration model."""
    models: ModelsConfig = ModelsConfig()
    endpoints: list[EndpointConfig] = Field(default_factory=lambda: [
        EndpointConfig(
            name="local",
            url="http://localhost:11434",
            roles=["reasoning", "tool_calling", "coding", "embedding", "importance"],
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
    model_settings: dict[str, dict] = Field(default_factory=dict)
