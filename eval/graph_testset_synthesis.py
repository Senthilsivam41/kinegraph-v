"""Neo4j-seeded graph-native synthetic question helpers (Phase 4).

RAGAS synthesizers do not traverse production Neo4j. This module builds
graph-local draft rows from verified entity/chunk MENTIONS when available,
falling back to offline entity-like tokens extracted from chunk text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z0-9_\-]{2,}(?:\s+[A-Z][A-Za-z0-9_\-]{2,}){0,3})\b")


@dataclass(frozen=True)
class GraphSeed:
    entity_name: str
    chunk_ids: tuple[str, ...]
    chunk_texts: tuple[str, ...]
    path_hint: str | None = None


def extract_offline_seeds(chunk_texts: Sequence[str], *, limit: int = 20) -> list[GraphSeed]:
    seeds: list[GraphSeed] = []
    seen: set[str] = set()
    for index, text in enumerate(chunk_texts):
        for match in ENTITY_RE.findall(text):
            name = match.strip()
            if name.lower() in seen or len(name) < 3:
                continue
            seen.add(name.lower())
            seeds.append(
                GraphSeed(
                    entity_name=name,
                    chunk_ids=(f"offline-chunk-{index}",),
                    chunk_texts=(text[:1200],),
                    path_hint=None,
                )
            )
            if len(seeds) >= limit:
                return seeds
    return seeds


def load_neo4j_seeds(driver: Any, *, limit: int = 50) -> list[GraphSeed]:
    """Load entities that have verified chunk MENTIONS."""
    query = """
    MATCH (e:Entity)<-[:MENTIONS]-(c)
    WHERE c:Chunk OR c:`__Chunk__`
    WITH e, collect(DISTINCT {id: c.id, text: c.text})[0..5] AS chunks
    WHERE size(chunks) > 0
    RETURN coalesce(e.name, e.id) AS name, chunks
    LIMIT $limit
    """
    seeds: list[GraphSeed] = []
    with driver.session() as session:
        for record in session.run(query, limit=limit):
            chunks = [item for item in record["chunks"] if item.get("id") and item.get("text")]
            if not chunks:
                continue
            seeds.append(
                GraphSeed(
                    entity_name=str(record["name"]),
                    chunk_ids=tuple(str(item["id"]) for item in chunks),
                    chunk_texts=tuple(str(item["text"]) for item in chunks),
                    path_hint="MENTIONS",
                )
            )
    return seeds


def synthesize_graph_rows(seeds: Iterable[GraphSeed], *, max_rows: int = 20) -> list[dict[str, Any]]:
    """Deterministic template questions grounded in seed chunk text (no invented facts)."""
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        if len(rows) >= max_rows:
            break
        context = seed.chunk_texts[0]
        rows.append(
            {
                "user_input": f"What does the knowledge graph evidence say about {seed.entity_name}?",
                "reference_contexts": repr([context]),
                "reference": (
                    f"According to verified source chunks linked to {seed.entity_name}, "
                    f"the supporting context states: {context[:400]}"
                ),
                "persona_name": "Graph Analyst",
                "query_style": "GRAPH_SEEDED",
                "query_length": "MEDIUM",
                "synthesizer_name": "graph_mentions_seed_v1",
                "graph_seed_entity": seed.entity_name,
                "graph_seed_chunk_ids": list(seed.chunk_ids),
                "graph_path_hint": seed.path_hint or "",
            }
        )
    return rows
