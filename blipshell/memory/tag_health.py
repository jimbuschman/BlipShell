"""Read-only tagging measurements shared by nightly and the database audit.

Tag counts describe coverage, not semantic correctness. In particular, leaving
the batch queue via _skip is not evidence that a memory acquired useful tags.
"""

TAG_HEALTH_SQL = """
WITH counts AS (
    SELECT mt.memory_id, COUNT(*) AS n,
           SUM(t.name NOT IN ('neutral', '_skip')) AS meaningful,
           MAX(t.name = '_skip') AS skipped,
           SUM(t.name = 'neutral') AS neutral
    FROM memory_tags mt JOIN tags t ON t.id = mt.tag_id
    GROUP BY mt.memory_id
)
SELECT COUNT(*) AS active,
       COALESCE(SUM(COALESCE(c.n, 0) = 0), 0) AS untagged,
       COALESCE(SUM(COALESCE(c.meaningful, 0) = 0), 0) AS without_topic_tags,
       COALESCE(SUM(c.n = 1 AND c.neutral = 1), 0) AS neutral_only,
       COALESCE(SUM(c.skipped = 1), 0) AS marked_skip,
       COALESCE(SUM(c.skipped = 1 AND COALESCE(c.meaningful, 0) <= 1), 0)
           AS skipped_low_coverage,
       COALESCE(SUM(m.summary IS NOT NULL AND COALESCE(c.n, 0) <= 1
                    AND COALESCE(c.skipped, 0) = 0), 0) AS pending
FROM memories m LEFT JOIN counts c ON c.memory_id = m.id
WHERE m.is_archived = 0
"""


def tag_health(connection) -> dict[str, int]:
    cursor = connection.execute(TAG_HEALTH_SQL)
    return dict(zip((d[0] for d in cursor.description), cursor.fetchone()))


async def tag_health_async(store) -> dict[str, int]:
    cursor = await store._db.execute(TAG_HEALTH_SQL)
    return dict(zip((d[0] for d in cursor.description), await cursor.fetchone()))


def tag_health_warnings(stats: dict) -> list[str]:
    """Absolute and proportional thresholds avoid alarms on tiny corpora."""
    warnings = []
    threshold = max(20, min(100, stats['active'] * 0.1))
    for key, label in (
        ('pending', 'memories awaiting batch tagging'),
        ('without_topic_tags', 'memories without tags other than neutral/skip'),
        ('neutral_only', 'memories tagged only neutral'),
        ('skipped_low_coverage', 'skip-marked memories with at most one non-placeholder tag'),
    ):
        if stats[key] >= threshold:
            warnings.append(f"{stats[key]} {label}")
    return warnings
