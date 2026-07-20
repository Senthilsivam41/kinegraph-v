"""Read-only ontology coverage analysis for a persisted graph and golden benchmark."""
from __future__ import annotations

import ast
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from backend.graph_ingestion.schema import OntologySchema


ENTITY_QUERY = """
MATCH (e) WHERE e:Entity OR e:`__Entity__`
WITH e, [label IN labels(e)
         WHERE NOT label IN ['Entity', '__Entity__', '__Node__']] AS domain_labels
RETURN coalesce(e.name, e.id) AS name,
       coalesce(e.type, e.label, head(domain_labels), 'Entity') AS entity_type
"""

RELATION_QUERY = """
MATCH (source)-[r]->(target)
WHERE (source:Entity OR source:`__Entity__`)
  AND (target:Entity OR target:`__Entity__`)
  AND type(r) <> 'MENTIONS'
WITH source, target, r,
     [label IN labels(source) WHERE NOT label IN ['Entity', '__Entity__', '__Node__']] AS source_labels,
     [label IN labels(target) WHERE NOT label IN ['Entity', '__Entity__', '__Node__']] AS target_labels
RETURN coalesce(source.type, source.label, head(source_labels), 'Entity') AS source_type,
       coalesce(r.type, type(r)) AS relation_type,
       coalesce(target.type, target.label, head(target_labels), 'Entity') AS target_type,
       coalesce(r.source, '') AS extraction_source
"""


def load_benchmark_rows(path: str | Path) -> list[dict[str, str]]:
    """Load question/reference text without assuming nonexistent claim/category fields."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as benchmark_file:
        for row in csv.DictReader(benchmark_file):
            contexts = row.get("reference_contexts") or ""
            try:
                parsed = ast.literal_eval(contexts)
                context_text = "\n".join(str(item) for item in parsed) if isinstance(parsed, list) else str(parsed)
            except (SyntaxError, ValueError):
                context_text = contexts
            rows.append({
                "question": row.get("user_input") or row.get("question") or "",
                "reference_text": f"{context_text}\n{row.get('reference') or ''}".strip(),
            })
    return rows


def load_graph_snapshot(driver: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the entity/relation type inventory from Neo4j without mutations."""
    with driver.session() as session:
        entities = [dict(record) for record in session.run(ENTITY_QUERY)]
        relationships = [dict(record) for record in session.run(RELATION_QUERY)]
    return entities, relationships


def audit_schema_coverage(
    schema: OntologySchema,
    benchmark_rows: list[dict[str, str]],
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare strict ontology coverage, fallback inventory, and golden references."""
    entity_counts = Counter(str(entity.get("entity_type") or "Entity") for entity in entities)
    relation_counts = Counter(str(rel.get("relation_type") or "RELATED_TO") for rel in relationships)
    fallback_counts = Counter(
        str(rel.get("relation_type") or "RELATED_TO")
        for rel in relationships
        if rel.get("extraction_source") == "simple_fallback"
    )
    out_of_schema_entities = {
        name: count for name, count in sorted(entity_counts.items())
        if name not in schema.entity_types and name != "Entity"
    }
    out_of_schema_relations = {
        name: count for name, count in sorted(relation_counts.items())
        if name not in schema.relation_types
    }

    valid_triples = sum(
        schema.validate_triple(
            str(rel.get("source_type") or "Entity"),
            str(rel.get("relation_type") or "RELATED_TO"),
            str(rel.get("target_type") or "Entity"),
        )
        for rel in relationships
    )
    entity_names = {
        str(entity.get("name")).strip().lower()
        for entity in entities
        if entity.get("name") and len(str(entity.get("name")).strip()) >= 3
    }
    uncovered_questions = []
    for row in benchmark_rows:
        reference = row["reference_text"].lower()
        if not any(name in reference for name in entity_names):
            uncovered_questions.append(row["question"])

    covered_rows = len(benchmark_rows) - len(uncovered_questions)
    recurring_fallback_relations = {
        name: count for name, count in sorted(fallback_counts.items()) if count >= 2
    }
    return {
        "schema_version": schema.version,
        "benchmark_rows": len(benchmark_rows),
        "graph_entities": len(entities),
        "graph_relationships": len(relationships),
        "golden_entity_mention_coverage": round(covered_rows / len(benchmark_rows), 4)
        if benchmark_rows else 0.0,
        "uncovered_questions": uncovered_questions,
        "strict_triple_coverage": round(valid_triples / len(relationships), 4)
        if relationships else 0.0,
        "schema_entity_types": list(schema.entity_types),
        "schema_relation_types": list(schema.relation_types),
        "observed_entity_types": dict(sorted(entity_counts.items())),
        "observed_relation_types": dict(sorted(relation_counts.items())),
        "out_of_schema_entity_types": out_of_schema_entities,
        "out_of_schema_relation_types": out_of_schema_relations,
        "recurring_fallback_relation_candidates": recurring_fallback_relations,
        "interpretation": (
            "Entity-mention coverage is a diagnostic proxy, not a RAGAS score. "
            "Review recurring fallback types and uncovered questions before changing the strict ontology."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the audit in a reviewable, versionable format."""
    lines = [
        "# Kinegraph Schema Coverage Audit",
        "",
        f"- Schema version: `{report['schema_version']}`",
        f"- Benchmark rows: **{report['benchmark_rows']}**",
        f"- Graph entities: **{report['graph_entities']}**",
        f"- Graph relationships: **{report['graph_relationships']}**",
        f"- Golden entity-mention coverage: **{report['golden_entity_mention_coverage']:.2%}**",
        f"- Strict valid-triple coverage: **{report['strict_triple_coverage']:.2%}**",
        "",
        "## Recurring fallback relation candidates",
        "",
    ]
    candidates = report["recurring_fallback_relation_candidates"]
    lines.extend([f"- `{name}`: {count}" for name, count in candidates.items()] or ["- None observed"])
    lines.extend(["", "## Golden questions without graph entity mentions", ""])
    lines.extend([f"- {question}" for question in report["uncovered_questions"]] or ["- None"])
    lines.extend(["", f"> {report['interpretation']}", ""])
    return "\n".join(lines)
