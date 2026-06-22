"""Model discovery — the shortlist layer.

Public leaderboards answer "which models exist and are they roughly good/cheap/
fast" — they do NOT predict how a model does on BlipShell's own prompts. So this
feeds the *discovery* step (what to bother testing), never the decision: the
harness + scoreboard make the actual switch call on the user's real tasks.

Two sources:
  * OpenRouter /api/v1/models — the model list needs no auth; gives id, context
    length, pricing, and input modalities (vision). BlipShell already routes
    through OpenRouter, so this is the natural candidate feed.
  * Artificial Analysis /api/v2 — Intelligence Index, tokens/sec, TTFT, price.
    Optional: skipped cleanly when no AA_API_KEY is set.

Parsers are pure (canned JSON -> normalized entries) so they're unit-testable
with no network. Fetchers wrap them with httpx and degrade gracefully on error.
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# AA v2 model dataset. If the path drifts, the fetch degrades gracefully (logged).
ARTIFICIAL_ANALYSIS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"


# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------

def _to_per_1m(per_token) -> Optional[float]:
    """OpenRouter prices are $/token as strings — convert to $/1M tokens."""
    if per_token is None:
        return None
    try:
        per_1m = round(float(per_token) * 1_000_000, 4)
    except (ValueError, TypeError):
        return None
    # OpenRouter uses negative sentinels for meta-routers (auto/fusion) — not a
    # real price; treat as unknown so they don't masquerade as "cheapest".
    return per_1m if per_1m >= 0 else None


def parse_openrouter(payload: dict, fetched_ts: str) -> list[dict]:
    """Normalize OpenRouter /models payload into catalog entries."""
    entries = []
    for m in (payload or {}).get("data", []) or []:
        pricing = m.get("pricing") or {}
        arch = m.get("architecture") or {}
        modalities = arch.get("input_modalities") or []
        if not modalities and arch.get("modality"):
            modalities = str(arch["modality"]).split("+")
        vision = any("image" in str(x).lower() for x in modalities)
        created = m.get("created")
        entries.append({
            "model": m.get("id") or m.get("canonical_slug") or m.get("name", ""),
            "source": "openrouter",
            "context_length": m.get("context_length") or m.get("top_provider", {}).get("context_length"),
            "price_in": _to_per_1m(pricing.get("prompt")),
            "price_out": _to_per_1m(pricing.get("completion")),
            "vision": vision,
            "intelligence_index": None,
            "tok_per_s": None,
            "ttft_s": None,
            "created_ts": str(created) if created is not None else None,
            "fetched_ts": fetched_ts,
            "raw": m,
        })
    return [e for e in entries if e["model"]]


def _first(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def parse_artificial_analysis(payload: dict, fetched_ts: str) -> list[dict]:
    """Normalize an Artificial Analysis v2 payload into catalog entries.

    Field names vary across AA index versions, so every field is looked up with
    fallbacks and the full record is retained in raw for drill-down.
    """
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(rows, dict):  # some shapes wrap the list deeper
        rows = rows.get("models") or rows.get("llms") or []
    entries = []
    for m in rows or []:
        pricing = m.get("pricing") or m
        model = _first(m, "slug", "id", "name", "model_name")
        if not model:
            continue
        entries.append({
            "model": model,
            "source": "artificial_analysis",
            "context_length": _first(m, "context_window", "context_length"),
            "price_in": _first(pricing, "price_1m_input_tokens", "price_input_1m", "price_in"),
            "price_out": _first(pricing, "price_1m_output_tokens", "price_output_1m", "price_out"),
            "vision": False,
            "intelligence_index": _first(
                m, "artificial_analysis_intelligence_index", "intelligence_index", "quality_index",
            ),
            "tok_per_s": _first(
                m, "median_output_tokens_per_second", "output_tokens_per_second", "tok_per_s",
            ),
            "ttft_s": _first(
                m, "median_time_to_first_token_seconds", "time_to_first_token_seconds", "ttft_s",
            ),
            "created_ts": _first(m, "release_date", "created"),
            "fetched_ts": fetched_ts,
            "raw": m,
        })
    return entries


# ---------------------------------------------------------------------------
# Async fetchers (network) — return [] and log on any failure.
# ---------------------------------------------------------------------------

async def fetch_openrouter(fetched_ts: str, api_key: Optional[str] = None, timeout: float = 20.0) -> list[dict]:
    headers = {}
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(OPENROUTER_MODELS_URL, headers=headers)
            resp.raise_for_status()
            return parse_openrouter(resp.json(), fetched_ts)
    except Exception as e:  # noqa: BLE001
        logger.warning("OpenRouter discovery failed: %s", e)
        return []


async def fetch_artificial_analysis(fetched_ts: str, api_key: Optional[str] = None, timeout: float = 20.0) -> list[dict]:
    key = api_key or os.environ.get("AA_API_KEY") or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
    if not key:
        logger.info("Artificial Analysis discovery skipped (no AA_API_KEY set)")
        return []
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(ARTIFICIAL_ANALYSIS_URL, headers={"x-api-key": key})
            resp.raise_for_status()
            return parse_artificial_analysis(resp.json(), fetched_ts)
    except Exception as e:  # noqa: BLE001
        logger.warning("Artificial Analysis discovery failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Shortlist
# ---------------------------------------------------------------------------

def shortlist(
    entries: list[dict],
    *,
    min_context: int = 0,
    max_price: float = 0.0,
    vision_only: bool = False,
    known_keys: Optional[set] = None,
) -> list[dict]:
    """Filter catalog entries to a candidate shortlist.

    - min_context: drop models with a smaller context window (0 = no floor).
    - max_price: drop models whose prompt price exceeds this $/1M (0 = no ceiling).
    - vision_only: keep only vision-capable models.
    - known_keys: set of (model, source) already seen; matching entries get
      is_new=False so the caller can highlight genuinely-new models.
    """
    known_keys = known_keys or set()
    out = []
    for e in entries:
        ctx = e.get("context_length") or 0
        if min_context and ctx and ctx < min_context:
            continue
        price = e.get("price_in")
        if max_price and price is not None and price > max_price:
            continue
        if vision_only and not e.get("vision"):
            continue
        e = dict(e)
        e["is_new"] = (e["model"], e["source"]) not in known_keys
        out.append(e)
    # New first, then cheapest, then biggest context.
    out.sort(key=lambda x: (
        not x["is_new"],
        x.get("price_in") if x.get("price_in") is not None else 1e9,
        -(x.get("context_length") or 0),
    ))
    return out
