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
    ranking: str = "qwen2.5:14b"
    importance: str = "qwen3:14b"
    ranking_importance: Optional[str] = None  # combined scoring; falls back to ranking model
    session_review: Optional[str] = None  # whole-session analysis; falls back to reasoning model
    reflection: Optional[str] = None  # idle self-reflection (lingering thoughts); falls back to reasoning model
    embedding: str = "qwen3-embedding:0.6b"
    # Fallback models used when cloud endpoints are unavailable
    reasoning_fallback: str = "gpt-oss:latest"
    tool_calling_fallback: str = "gpt-oss:latest"
    coding_fallback: str = "gpt-oss:latest"
    summarization_fallback: Optional[str] = None
    ranking_fallback: Optional[str] = None
    importance_fallback: Optional[str] = None
    ranking_importance_fallback: Optional[str] = None
    session_review_fallback: Optional[str] = None
    reflection_fallback: Optional[str] = None
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
    rate_limit_tpm: Optional[int] = None  # max tokens per minute (prevents TPM 429s)
    models: dict[str, str] = Field(default_factory=dict)  # per-endpoint model overrides
    pii_sanitize: Optional[bool] = None  # None = auto (true for openai, false for ollama)
    cost_per_1m_prompt: float = 0.0  # $/million prompt tokens (0 = free/local)
    cost_per_1m_completion: float = 0.0  # $/million completion tokens (0 = free/local)


class PoolConfig(BaseModel):
    """Configuration for a memory token budget pool."""
    percentage: float
    max_tokens: Optional[int] = None
    max_items: Optional[int] = None  # Hard cap on number of items (None = unlimited)
    priority: int = 0


class MemoryPoolsConfig(BaseModel):
    """All memory pool configurations.

    Pool contracts:
    - Core: Stable personal facts (core_memories table). Always present. Small.
    - Lessons: Top extracted insights. Always present. Capped at 30.
    - ActiveSession: Current conversation messages.
    - RecentHistory: Previous session memories + summaries.
    - Recall: Search results — most relevant content for the current query. Largest pool.
    """
    core: PoolConfig = PoolConfig(percentage=0.05, priority=5, max_items=20)
    lessons: PoolConfig = PoolConfig(percentage=0.05, priority=4, max_items=30)
    active_session: PoolConfig = PoolConfig(percentage=0.30, priority=3)
    recent_history: PoolConfig = PoolConfig(percentage=0.20, priority=2)
    recall: PoolConfig = PoolConfig(percentage=0.40, priority=1)
    # Buffer removed — was always empty, wasted 10% of budget.
    # Legacy config.yaml with buffer key will be ignored (Pydantic extra="ignore").

    def model_post_init(self, __context):
        """Enforce pool item caps even when config.yaml overrides lose defaults."""
        if self.core.max_items is None:
            self.core.max_items = 20
        if self.lessons.max_items is None:
            self.lessons.max_items = 30


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
    importance_recency_bonus: float = 0.1
    importance_tag_bonus: float = 0.05
    similarity_threshold: float = 0.35  # industry standard for semantic search (was 0.5, too strict at scale)
    min_importance: float = 0.25  # replaces min_rank filter — continuous, better predictor of recall relevance
    fts_baseline_similarity: float = 0.4  # FTS-only hits get this baseline so they can compete on other signals
    importance_boost_weight: float = 0.2
    tag_overlap_boost: float = 0.1
    search_overfetch_multiplier: int = 2
    # Time-aware search: parse time expressions in queries ("yesterday", "last
    # week") and rank in-range memories first. Deterministic regex on the
    # per-turn path — see memory/timeparse.py for the evidence and the trade.
    time_aware_search: bool = True
    decay_rate: float = 0.001  # temporal decay rate (~50% after 29 days) — global fallback

    fts_weight: float = 0.3  # weight for FTS5 RRF boost in hybrid search
    auto_prune_days: int = 0  # 0 = disabled; was 90 but archived 1083 imported memories
    prune_max_importance: float = 0.3
    prune_max_rank: int = 2
    # Entity graph pruning (soft-archive low-value entities; nightly "prune_entities" job)
    entity_prune_enabled: bool = False  # opt-in — off by default
    entity_prune_dry_run: bool = True   # when enabled, log-only until explicitly set False
    entity_prune_min_age_days: int = 30  # never prune entities younger than this
    entity_prune_max_mentions: int = 1   # prune only if mention count <= this
    entity_prune_max_degree: int = 1     # prune only if relationship degree <= this
    # Retroactive entity merge (consolidate existing duplicates; nightly "merge_entities").
    # Conservative on purpose — a bad merge destroys edge differentiation. Higher
    # thresholds than creation-time resolution (which is 0.85 auto / 0.70 LLM).
    entity_merge_enabled: bool = False        # opt-in
    entity_merge_dry_run: bool = True         # log-only until explicitly set false
    entity_merge_auto_threshold: float = 0.90  # auto-merge above this similarity
    entity_merge_llm_threshold: float = 0.80   # LLM arbitration band: this..auto
    entity_merge_max_candidates: int = 5       # similar entities checked per entity
    entity_merge_edge_sample: int = 5          # edges shown to the arbitration LLM
    # 0.92 not 0.85: measured on the real 17K corpus 2026-08-06. At 0.85 the
    # 0.85-0.88 band merged semantically unrelated memories, because a vague
    # summary is close to everything. See config.yaml for examples.
    consolidation_similarity: float = 0.92
    # Memories checked per nightly run (0 = disabled). Was 20, which against a
    # 17K corpus meant ~14 months for a single sweep. Raised once the per-check
    # Ollama round trip was removed — checks are now a local vector scan, so
    # the real limit is the job's time budget, not the count.
    # 500: measured 7.3 memories/sec on a 17K corpus (2026-08-06), so a
    # 500-batch scans in ~70s and fits comfortably inside the scan's share
    # of the 270s job budget. 2000 did not — it spent the whole budget
    # scanning and processed nothing. Raise only with a measurement.
    consolidation_batch_size: int = 500
    # Log merges without applying them. Worth setting true for the first run
    # at the new throughput: a bad threshold merges far more at 2000/night
    # than it ever could at 20.
    consolidation_dry_run: bool = False
    contradiction_similarity_threshold: float = 0.7  # min similarity to check for contradiction
    tag_discovery_interval_days: int = 7  # days between discovery runs
    tag_discovery_sample_size: int = 20  # poorly-tagged memories to sample
    entity_extraction_batch_size: int = 50  # memories processed per startup run
    entity_boost: float = 0.15  # boost for memories found via entity graph
    project_boost: float = 0.15  # boost for memories from active project sessions
    recency_boost_weight: float = 0.15  # max recency boost amplitude
    # FadeMem: importance-modulated decay (replaces flat 48h half-life)
    fadem_enabled: bool = False  # off by default — flat 48h decay is safer on old corpora
    fadem_base_rate: float = 0.001  # base decay rate (~29 day half-life, modulated by importance)
    fadem_importance_factor: float = 2.0  # how much importance slows decay (higher = slower for imp=1.0)
    fadem_access_hours: float = 24.0  # hours subtracted per access_count (strengthening)
    dedup_jaccard_threshold: float = 0.65  # Jaccard similarity to consider summaries duplicate

    # Lesson lifecycle (2026-09-02 audit follow-up) — the store's two decay
    # modes were unbounded accumulation of paraphrase-duplicate lessons and
    # once-at-birth importance scores that no evidence ever revisited.
    # Family folding: nightly, inside clean_junk_lessons. STRICTER than the
    # audit's 0.20 measuring threshold — the hand-reviewed one-shot found 6
    # false families in 127 at 0.20; an unattended job only merges blatant
    # paraphrases. 0 disables.
    lesson_family_fold_threshold: float = 0.35
    # Revoting: nightly job pairs fresh session reflections with the most
    # similar lessons and asks the LOCAL model confirms/contradicts; votes
    # move importance (down harder than up), and the lessons pool's top-30
    # cut does the rest. Demotion only, never deletion. Ships OFF + dry-run:
    # an LLM auto-editing a permanent pool earns trust through its dry-run
    # reports first (same pattern as entity_merge).
    lesson_revote_enabled: bool = False
    lesson_revote_dry_run: bool = True
    lesson_revote_up: float = 0.05
    lesson_revote_down: float = 0.15
    lesson_revote_per_reflection: int = 3   # lessons judged per new reflection
    lesson_revote_max_pairs: int = 40       # per-night cap (fits the job budget)
    project_session_limit: int = 50  # max recent project sessions for two-pass search
    centroid_tag_similarity: float = 0.75  # cosine similarity threshold for centroid tag assignment
    centroid_tag_min_members: int = 10  # minimum tagged memories to compute a tag centroid
    centroid_tag_batch_size: int = 500  # memories processed per centroid tagging batch
    batch_tag_batch_size: int = 10  # memory summaries per LLM batch tag call
    batch_tag_max_batches: int = 500  # max LLM batches per nightly run
    dedup: DedupConfig = DedupConfig()
    entity_resolution: EntityResolutionConfig = EntityResolutionConfig()
    # Reranker — rescores top search results using a cross-encoder model
    reranker_enabled: bool = False  # off by default (requires model pulled on Ollama)
    reranker_model: str = "dengcao/Qwen3-Reranker-0.6B:Q8_0"
    reranker_top_n: int = 15  # rerank this many candidates, return all (sorted by reranker score)
    reranker_weight: float = 0.4  # blend: (1-w)*boosted_score + w*reranker_score
    reranker_instruction: str = ""  # custom instruction (empty = use default)


class SessionConfig(BaseModel):
    """Session management configuration."""
    max_messages_before_summary: int = 50
    summary_chunk_size: int = 20
    auto_save_interval: int = 300


class AgentConfig(BaseModel):
    """Agent behavior configuration."""
    max_tool_iterations: int = 50  # CC has no limit; 50 is a generous safety net
    system_prompt: str = (
        "You are BlipShell, a local AI assistant with persistent memory.\n"
        "Past conversations and lessons are automatically loaded into your context.\n"
        "You support image input: when the user attaches an image, BlipShell sends "
        "it to you directly and (on vision-capable models) you can see it — describe "
        "what is actually in the image. Never claim BlipShell can't receive or handle "
        "images; that pipeline exists. If you genuinely cannot see an attached image, "
        "the active model isn't vision-capable — say that specifically.\n\n"
        "# Rules\n"
        "1. Always answer the user's most recent message directly. Do not drift into "
        "the broader conversation topic — focus on exactly what was asked.\n"
        "2. Be concise and direct. Do NOT narrate your process — just do it or answer.\n"
        "3. When the user references past conversations, events, or asks 'do you remember', "
        "ALWAYS use search_memories to find the relevant details. Your loaded context is a "
        "summary — search for specifics before saying you don't know.\n"
        "4. Only use file/code tools when genuinely needed. Answer general questions from context.\n"
        "5. Read a file before editing it. Make minimal changes — no extras.\n"
        "6. If something fails twice, ask the user instead of retrying blindly.\n"
        "7. Each tool's description explains when to use it — follow that guidance.\n"
    )
    stream: bool = True
    # Tool approval: tools listed here require user confirmation before execution
    tools_requiring_approval: list[str] = Field(default_factory=lambda: [
        "write_file", "edit_file", "run_command", "git_add", "git_commit",
        "activate_project", "deactivate_project",
    ])
    auto_approve_tools: bool = False  # bypass approval for all tools


class ShellToolConfig(BaseModel):
    """Shell tool configuration."""
    timeout: int = 30
    allowed_commands: list[str] = Field(default_factory=list)  # empty = allow all (CC approach)


class FilesystemToolConfig(BaseModel):
    """Filesystem tool configuration."""
    max_file_size: int = 1048576
    blocked_paths: list[str] = Field(default_factory=lambda: ["/etc/shadow", "/etc/passwd"])


class WebToolConfig(BaseModel):
    """Web tool configuration."""
    max_fetch_size: int = 524288
    timeout: int = 15
    tavily_api_key: Optional[str] = None  # Tavily search API key, supports ${ENV_VAR}


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
    embedding_dimensions: int = 1024  # qwen3-embedding:0.6b output size
    # Refuse to run when the database file does not exist. An absent SQLite
    # file is a CREATION, not a failure — which is how 2026-08-11..08-20 put
    # nine days of live sessions into a phantom 16MB database while the real
    # 491MB corpus sat untouched, with no symptom but an assistant that had
    # quietly lost its history. On an instance whose corpus already exists,
    # a missing file at the resolved path is ALWAYS a wrong-path launch, so
    # production config sets this true. Default false so fresh installs and
    # temp-DB test configs keep working.
    require_existing: bool = False


class LLMConfig(BaseModel):
    """LLM call configuration."""
    max_retries: int = 2
    retry_base_delay: float = 1.0
    timeout: float = 120.0  # per-call timeout in seconds


class AuthConfig(BaseModel):
    """Web UI authentication configuration."""
    enabled: bool = False
    api_key: str = ""


class TelegramConfig(BaseModel):
    """Telegram bot configuration."""
    bot_token: str = ""  # from @BotFather, supports ${ENV_VAR}
    allowed_user_ids: list[int] = Field(default_factory=list)  # empty = allow all (DANGEROUS)
    enabled: bool = False


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


class PIIConfig(BaseModel):
    """PII sanitization configuration."""
    enabled: bool = True  # sanitize PII before cloud calls
    # Start sessions in local mode (/local): endpoints that relay off the
    # machine are invisible to routing until /cloud. Off by default — the
    # cloud chat model is the reason it's configured; local mode is the
    # explicit per-conversation privacy choice (V2_PLAN D1).
    local_mode_default: bool = False


class RoboticsConfig(BaseModel):
    """Modular cube robotics — software-first transport server.

    When enabled, BlipShell listens for cube connections (in-process virtual
    cubes or the standalone window over a socket). Cubes auto-register on
    connect; their actions become tools and the LLM authors behaviors. Disabled
    by default — opening a listening socket is opt-in.
    """
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    invoke_timeout: float = 10.0  # seconds to await a remote cube's action result
    idle_seconds: float = 30.0    # quiet time before a truthful system_idle event fires


class ReflectionConfig(BaseModel):
    """Idle self-reflection — BlipShell forms a self-originated 'lingering
    thought' after a long quiet gap, then may raise it unprompted on return.

    Experimental (toward genuine continuity). Fires once per idle gap (re-arms
    on activity), generated from its own prior thoughts only (no transcript).
    """
    enabled: bool = True
    idle_seconds: float = 10800.0   # ~3 hours of quiet before it reflects
    max_keep: int = 50              # how many past thoughts to retain
    # Reflect at startup when the gap since the last session's activity
    # exceeds idle_seconds: the quiet time genuinely happened, the process
    # just wasn't awake to notice. Without this, thoughts only form when the
    # app sits OPEN and idle 3+ hours — on open-chat-close usage that's
    # ~1 thought/month, which made the self-gravity step-2 gate ("10 new
    # thoughts") roughly a year away (2026-08-09 analysis).
    on_return_enabled: bool = True

    # Nightly reflection: form one lingering thought during the nightly run.
    # The idle loop and on-return reflection both require the process to be
    # (or have been) around a 3h+ gap; measured throughput on open-chat-close
    # usage was ~1 thought/month, and the self-gravity step-2 gate needs 10
    # NEW thoughts — a year away at that rate (scripts/diagnose_self_thoughts).
    # The nightly run happens regardless of app usage, so it is the one place
    # a steady cadence can live. Guarded by nightly_min_gap_hours so a night
    # following a same-day idle/return thought doesn't pile on a second one.
    nightly_enabled: bool = True
    nightly_min_gap_hours: float = 12.0

    # Standing injection: a relevant past thought resurfaces as context (not just
    # the one-shot greeting). Two-stage filter — cosine prefilter then an LLM
    # relevance judge (local reasoning model, yes/no). Gated by inject_enabled
    # only; does NOT use the reranker. Fail-closed: if the judge errors, nothing
    # injects. (The judge replaced a Qwen3 reranker that didn't produce usable
    # output via Ollama — see search.py search_self_thoughts.)
    inject_enabled: bool = True
    inject_cosine_floor: float = 0.4   # loose recall prefilter — NOT the gate
    inject_rerank_floor: float = 0.8   # the gate: judge verdict (1.0/0.0) must clear this
    inject_max: int = 1                # max thoughts injected per turn (backstop)
    inject_prefilter_k: int = 3        # candidates handed to the judge

    # Self-gravity (step 1): each thought carries a weight that recurrence
    # reinforces and surfacing/age decay erodes. Among thoughts that pass the
    # relevance gate, the heaviest (weight x relevance) surfaces, and heavy ones
    # render with a "recurring" marker. OFF by default — graduated opt-in. This
    # lives entirely in the self-layer: it never touches retrieval ranking, so
    # the assistant cannot regress. The relevance gate is unchanged and still
    # required, so gravity only re-orders/marks what was already going to surface.
    gravity_enabled: bool = False
    gravity_recur_threshold: float = 0.85  # cosine above which a new thought "echoes" a prior
    gravity_recur_boost: float = 0.5       # weight added to an echoed prior (recurrence = gravity)
    gravity_fatigue: float = 0.6           # weight multiplier each time a thought surfaces (anti-spiral)
    gravity_half_life_days: float = 30.0   # age decay: effective weight halves over this span
    gravity_min_weight: float = 0.1        # floor so decay/fatigue never fully zero a thought
    gravity_marker_weight: float = 1.5     # render "recurring" marker at/above this effective weight


class CompactionConfig(BaseModel):
    """Structured compaction configuration.

    When enabled, uses LLM-driven 9-section summary to compress older
    conversation context while preserving key information. Falls back
    to mechanical per-tool-type compression on LLM failure.
    """
    enabled: bool = False                       # disabled by default — enable after testing on Ollama PC
    use_llm: bool = True                        # LLM-driven summary; False = mechanical only
    compaction_threshold: float = 0.95           # trigger at this fraction of context_limit (was 0.85 — too aggressive)
    partial_compaction: bool = True              # only summarize old messages, keep recent verbatim
    min_recent_user_messages: int = 5            # keep at least this many recent user messages
    min_recent_tokens: int = 10000               # keep at least this many tokens of recent conversation
    file_restoration: bool = True                # re-inject recently-read files post-compaction
    max_restore_files: int = 5                   # max files to restore
    max_restore_tokens_per_file: int = 5000      # max tokens per restored file
    max_restore_tokens_total: int = 25000        # total token cap for all restored files
    summary_timeout: float = 60.0                # timeout for the LLM compaction call


class NotesConfig(BaseModel):
    """Session notes configuration.

    Session notes are persistent key-value pairs that survive context
    compaction. Both the LLM (via tools) and user (via /notes) can
    manage them. Stored in sessions.metadata_json.
    """
    enabled: bool = True
    max_notes: int = 50                          # max number of notes per session
    max_total_tokens: int = 4000                 # total token budget for all notes (was 12K — too much context pressure)
    max_note_tokens: int = 2000                  # per-note token limit


class GuardrailsConfig(BaseModel):
    """Toggleable guardrails for instruction adherence.

    When enabled, adds mid-execution checks to reduce specification drift,
    forgotten requirements, and repeated mistakes.
    """
    enabled: bool = True
    completion_audit: bool = True       # re-check original request before accepting task_complete
    correction_detector: bool = True    # detect user corrections → anti-pattern lessons
    correction_judge: bool = True       # confirm regex candidates with the local LLM before minting (fail-closed)
    trajectory_monitor: bool = True     # periodic state injection with original task reminder
    context_pinning: bool = True        # pin original task in compaction
    requirement_checklist: bool = True  # confirm_plan tool before execution
    monitor_interval: int = 5           # inject trajectory check every N tool calls
    max_audit_retries: int = 2          # max times to reject task_complete before accepting
    completion_audit_min_tool_calls: int = 5  # below this (and no checklist/multi-file), skip the LLM audit — trivial tasks don't need it
    # Look-before-review — ground review/critique requests in a real read/grep.
    # Cheap: prompt guidance (both chat paths) + a zero-LLM completion gate that
    # refuses task_complete when a review finishes with no read/grep this turn.
    review_grounding: bool = True       # enforce look-before-review on review requests
    # Doom-loop detector — cheap counter-based pattern detection (no LLM cost)
    doom_loop_detector: bool = True     # detect repetitive/stuck behavior patterns
    doom_loop_read_threshold: int = 3   # warn after reading same file N times
    doom_loop_edit_threshold: int = 3   # warn after editing same file N times
    doom_loop_readonly_streak: int = 8  # warn after N consecutive read-only tools with no writes


class BenchmarkConfig(BaseModel):
    """Unified model-benchmark harness configuration.

    Drives `blipshell benchmark` — running candidate models through the
    existing benchmark suites, grading open-ended outputs with a neutral
    cloud judge, and rendering a switch-verdict vs the current production
    models. All optional: an absent `benchmark:` block uses these defaults.
    """
    db_path: str = "data/benchmark.db"  # dedicated store; NOT the production memory DB
    # Neutral cloud judge — grades open-ended tasks (summarization/reasoning/lessons)
    # 0..1. Must name an endpoint present in `endpoints:`. Empty judge_model = no judging
    # (deterministic metrics only). The judge model is excluded from candidate runs.
    judge_model: str = ""
    judge_endpoint: str = ""  # endpoint name from `endpoints:` to route the judge through
    judge_timeout: float = 60.0  # per-judge-call timeout (asyncio.wait_for)
    # Verdict: candidate is "better"/"worse" only when the per-task delta vs the
    # production baseline exceeds this; otherwise "tie".
    verdict_delta: float = 0.05
    # Composite weighting per task type (empty = equal weight across measured tasks).
    task_weights: dict[str, float] = Field(default_factory=dict)
    # Tier sample sizes.
    quick_sample: int = 8    # synthetic curated messages (pipeline tier)
    full_sample: int = 50    # real-DB sample for the full tier
    # Discovery: shortlist defaults for `benchmark discover`.
    discover_min_context: int = 0       # 0 = no floor
    discover_max_price: float = 0.0     # 0 = no ceiling ($/1M prompt tokens)


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
    telegram: TelegramConfig = TelegramConfig()
    web_ui: WebUIConfig = WebUIConfig()
    pii: PIIConfig = PIIConfig()
    planner: PlannerConfig = PlannerConfig()
    compaction: CompactionConfig = CompactionConfig()
    notes: NotesConfig = NotesConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()
    worker: WorkerConfig = WorkerConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    robotics: RoboticsConfig = RoboticsConfig()
    reflection: ReflectionConfig = ReflectionConfig()
    model_settings: dict[str, dict] = Field(default_factory=dict)
