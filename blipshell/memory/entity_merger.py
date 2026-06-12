"""Retroactive entity merge — consolidate duplicate entities already in the graph.

EntityExtractor._resolve_entity dedups at CREATION time, when the surrounding
memory gives context. This pass runs over entities ALREADY stored (e.g.
duplicates created while nightly extraction was running without resolution),
where the only signal is the entity's own relationships.

Conservative by design. A bad merge collapses two distinct entities and destroys
their edge differentiation — harder to recover than restoring a soft-archived
node. So vs. creation-time resolution this uses HIGHER thresholds by default
(0.90 auto-merge, 0.80-0.90 LLM-arbitration band instead of 0.85 / 0.70-0.85),
the arbitration prompt is given each entity's edge sample, and the prompt is
biased toward NO when evidence is weak.

Each merge: reassign edges/mentions to the canonical (more-mentioned) entity via
merge_entity, record an alias for the audit trail, and soft-archive the
merged-away husk (reversible). Dry-run reports the plan without touching data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from blipshell.llm.prompts import resolve_entity_merge_with_edges
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore

if TYPE_CHECKING:
    from blipshell.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EntityMerger:
    """Finds and merges duplicate entities already stored in the graph."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        router: LLMRouter,
        vectors: VectorStore,
        *,
        auto_threshold: float = 0.90,
        llm_threshold: float = 0.80,
        max_candidates: int = 5,
        edge_sample: int = 5,
        task_type: str = TaskType.REASONING,
    ):
        self.sqlite = sqlite
        self.router = router
        self.vectors = vectors
        self.auto_threshold = auto_threshold
        self.llm_threshold = llm_threshold
        self.max_candidates = max_candidates
        self.edge_sample = edge_sample
        self.task_type = task_type

    @staticmethod
    def _pick_canonical(a: dict, b: dict) -> tuple[dict, dict]:
        """Return (keep, drop). Keeper = more mentions; tie broken by lower id."""
        if (a["mentions"], -a["id"]) >= (b["mentions"], -b["id"]):
            return a, b
        return b, a

    async def _llm_confirms_merge(self, a: dict, b: dict) -> bool:
        """Edge-aware LLM arbitration for the ambiguous similarity band."""
        try:
            edges_a = await self.sqlite.get_entity_edge_sample(
                a["id"], a["name"], self.edge_sample,
            )
            edges_b = await self.sqlite.get_entity_edge_sample(
                b["id"], b["name"], self.edge_sample,
            )
            system, prompt = resolve_entity_merge_with_edges(
                a["name"], edges_a, b["name"], edges_b,
            )
            response = await self.router.generate(
                self.task_type, prompt, system=system, think=False,
            )
            return response.strip().upper().startswith("YES")
        except Exception as e:
            # Conservative: any failure means "don't merge".
            logger.warning("Merge arbitration failed for '%s'/'%s': %s",
                           a["name"], b["name"], e)
            return False

    async def merge_pass(self, dry_run: bool = True) -> dict:
        """Run one retroactive merge pass. Returns stats (+ a plan sample).

        When dry_run is True, computes the full merge plan and logs it but makes
        no changes. When False, applies the plan: merge_entity → record alias →
        soft-archive the merged-away entity.
        """
        entities = await self.sqlite.get_mergeable_entities()
        by_id = {e["id"]: e for e in entities}
        merged_away: set[int] = set()
        plan: list[dict] = []
        stats = {
            "entities_scanned": len(entities),
            "auto_merges": 0,
            "llm_merges": 0,
            "llm_rejects": 0,
        }

        for e in entities:
            if e["id"] in merged_away:
                continue
            try:
                candidates = self.vectors.search_similar_entities(
                    e["name"], n_results=self.max_candidates,
                )
            except Exception as ex:
                logger.warning("Similarity search failed for '%s': %s", e["name"], ex)
                continue

            for c in candidates:
                cid = c.get("id")
                sim = c.get("similarity", 0.0)
                if cid == e["id"] or cid in merged_away or cid not in by_id:
                    continue
                # Candidates are returned most-similar first — once we drop below
                # the arbitration floor, nothing further is worth checking.
                if sim < self.llm_threshold:
                    break

                other = by_id[cid]
                method = None
                if sim >= self.auto_threshold:
                    method = "retroactive_embedding"
                    stats["auto_merges"] += 1
                elif await self._llm_confirms_merge(e, other):
                    method = "retroactive_llm"
                    stats["llm_merges"] += 1
                else:
                    stats["llm_rejects"] += 1
                    continue

                keep, drop = self._pick_canonical(e, other)
                plan.append({
                    "drop_id": drop["id"], "drop_name": drop["name"],
                    "keep_id": keep["id"], "keep_name": keep["name"],
                    "similarity": round(sim, 4), "method": method,
                })
                merged_away.add(drop["id"])
                # If e itself was merged away, stop scanning its candidates.
                if drop["id"] == e["id"]:
                    break

        # Sort the plan by confidence (highest similarity first) so a human
        # reviewing the dry-run reads the safest merges first and the borderline
        # LLM-band ones last — that tail is where to look hardest.
        plan.sort(key=lambda p: -p["similarity"])
        sample = plan[:25]
        if dry_run:
            for p in sample:
                logger.info(
                    "[dry-run] would merge '%s' (id=%d) -> '%s' (id=%d) "
                    "sim=%.3f via %s",
                    p["drop_name"], p["drop_id"], p["keep_name"], p["keep_id"],
                    p["similarity"], p["method"],
                )
            # Return the FULL plan (sorted) so the caller can persist a complete,
            # reviewable preview — the 25-row sample isn't enough on a large graph.
            return {**stats, "dry_run": True, "would_merge": len(plan),
                    "sample": sample, "plan": plan}

        applied = 0
        for p in plan:
            try:
                await self.sqlite.merge_entity(p["drop_id"], p["keep_id"])
                await self.sqlite.record_entity_alias(
                    p["drop_name"], p["keep_id"], merge_method=p["method"],
                )
                await self.sqlite.archive_entities([p["drop_id"]])
                if self.vectors:
                    try:
                        self.vectors.delete_entity(p["drop_id"])
                    except Exception:
                        pass  # orphan vector, swept later
                applied += 1
            except Exception as ex:
                logger.warning("Failed to merge id=%d into id=%d: %s",
                               p["drop_id"], p["keep_id"], ex)
        return {**stats, "dry_run": False, "merged": applied,
                "planned": len(plan), "sample": sample}
