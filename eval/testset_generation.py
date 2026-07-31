"""Governed RAGAS TestsetGenerator pipeline for Kinegraph benchmarks.

Uses ADR-002 ``chunk_document`` + RAGAS ``generate_with_chunks`` so synthetic
questions are grounded in production-shaped chunks. Outputs are versioned drafts
by default and never silently overwrite an accepted benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document

from backend.graph_ingestion.adaptive_chunking import (
    CHUNK_POLICY_VERSION,
    ChunkRecord,
    chunk_document,
)
from eval.benchmark_reference_audit import build_draft_audit, write_reference_audit
from eval.experiment_validation import current_git_revision, sha256_file
from eval.usage_cost import UsageTracker


def generate_document_id(file_path: str) -> str:
    digest = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
    return f"doc_{digest}"


REQUIRED_COLUMNS = (
    "user_input",
    "reference_contexts",
    "reference",
    "persona_name",
    "query_style",
    "query_length",
    "synthesizer_name",
)

DEFAULT_DISTRIBUTION = (
    ("single_hop_specific", 0.4),
    ("multi_hop_specific", 0.4),
    ("multi_hop_abstract", 0.2),
)


@dataclass(frozen=True)
class GenerationResult:
    csv_path: Path
    audit_path: Path
    manifest_path: Path
    note_path: Path | None
    row_count: int
    elapsed_seconds: float
    chunk_policy_version: str
    adaptive_enabled: bool
    usage: dict[str, Any] = field(default_factory=dict)


def load_markdown_documents(docs_dir: str | Path) -> list[Document]:
    docs_dir = Path(docs_dir)
    documents: list[Document] = []
    if not docs_dir.is_dir():
        return documents
    for path in sorted(docs_dir.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={"source": str(path.as_posix()), "file_name": path.name},
            )
        )
    return documents


def records_to_langchain_docs(records: Sequence[ChunkRecord]) -> list[Document]:
    docs: list[Document] = []
    for record in records:
        meta = record.to_metadata()
        meta["chunk_id"] = record.chunk_id
        meta["source"] = record.document_id
        docs.append(Document(page_content=record.text, metadata=meta))
    return docs


def chunk_documents_for_generation(
    documents: Sequence[Document],
    *,
    adaptive_enabled: bool,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for document in documents:
        source = str(document.metadata.get("source") or "doc_unknown")
        doc_id = generate_document_id(source)
        records.extend(
            chunk_document(
                document.page_content,
                document_id=doc_id,
                adaptive_enabled=adaptive_enabled,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return records


def _build_query_distribution(llm: Any) -> list[tuple[Any, float]]:
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    mapping = {
        "single_hop_specific": SingleHopSpecificQuerySynthesizer(llm=llm),
        "multi_hop_specific": MultiHopSpecificQuerySynthesizer(llm=llm),
        "multi_hop_abstract": MultiHopAbstractQuerySynthesizer(llm=llm),
    }
    return [(mapping[name], weight) for name, weight in DEFAULT_DISTRIBUTION]


def _normalize_dataframe(df: Any) -> Any:
    import pandas as pd

    renamed = {
        "question": "user_input",
        "contexts": "reference_contexts",
        "ground_truth": "reference",
        "answer": "reference",
    }
    frame = df.rename(columns={key: value for key, value in renamed.items() if key in df.columns})
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            if column == "persona_name":
                frame[column] = "unknown"
            elif column == "query_style":
                frame[column] = "SYNTHETIC"
            elif column == "query_length":
                frame[column] = "MEDIUM"
            elif column == "synthesizer_name":
                frame[column] = "unknown"
            else:
                frame[column] = ""
    # Persist list-valued contexts as literal strings for CSV round-trip.
    if "reference_contexts" in frame.columns:
        frame["reference_contexts"] = frame["reference_contexts"].apply(
            lambda value: value if isinstance(value, str) else repr(list(value or []))
        )
    return frame[list(REQUIRED_COLUMNS)]


def build_generation_manifest(
    *,
    repo_root: Path,
    csv_path: Path,
    audit_path: Path,
    row_count: int,
    adaptive_enabled: bool,
    chunk_policy_version: str,
    generation_model: str,
    embedding_model: str,
    testset_size: int,
    elapsed_seconds: float,
    usage: dict[str, Any],
    docs_dir: str,
) -> dict[str, Any]:
    return {
        "schema_version": "kinegraph.testset.generation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": current_git_revision(repo_root),
        "docs_dir": docs_dir,
        "testset_size_requested": testset_size,
        "row_count": row_count,
        "chunk_policy_version": chunk_policy_version,
        "adaptive_chunking": adaptive_enabled,
        "synthesizer_distribution": [
            {"name": name, "weight": weight} for name, weight in DEFAULT_DISTRIBUTION
        ],
        "models": {
            "generation": generation_model,
            "embedding": embedding_model,
            "scoring_embedding_identity": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "artifacts": {
            "dataset_csv": str(csv_path.relative_to(repo_root)),
            "draft_audit": str(audit_path.relative_to(repo_root)),
            "dataset_sha256": sha256_file(csv_path),
        },
        "timing": {"elapsed_seconds": round(elapsed_seconds, 3)},
        "usage": usage,
        "cost_complete": bool(usage.get("cost_complete", False)),
    }


def write_spike_note(
    path: Path,
    *,
    result: GenerationResult,
    synthesizer_counts: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Synthetic testset generation spike",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Rows: {result.row_count}",
        f"- Elapsed seconds: {result.elapsed_seconds:.3f}",
        f"- Chunk policy: {result.chunk_policy_version}",
        f"- Adaptive enabled: {result.adaptive_enabled}",
        f"- CSV: `{result.csv_path}`",
        f"- Draft audit: `{result.audit_path}`",
        f"- Manifest: `{result.manifest_path}`",
        f"- Usage: `{json.dumps(result.usage)}`",
        "",
        "## Synthesizer mix observed",
        "",
    ]
    for name, count in sorted(synthesizer_counts.items()):
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Generation embeddings use OpenRouter-compatible OpenAI embeddings.",
            "- Live RAGAS scoring continues to use local MiniLM for eval identity.",
            "- Output is a draft until `scripts/audit_benchmark_references.py --accept` is run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_testset(
    *,
    repo_root: str | Path,
    docs_dir: str | Path = "docs",
    size: int = 12,
    output_version: str = "v1.2.0-draft",
    adaptive_enabled: bool | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    generation_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    force_new_version: bool = False,
    write_spike_note_file: bool = True,
    dry_run_chunks_only: bool = False,
) -> GenerationResult:
    """Generate a versioned draft benchmark CSV + draft audit + generation manifest."""
    repo_root = Path(repo_root).resolve()
    docs_dir_path = (repo_root / docs_dir).resolve() if not Path(docs_dir).is_absolute() else Path(docs_dir)
    if adaptive_enabled is None:
        from backend.core.config import settings

        adaptive_enabled = bool(settings.ADAPTIVE_CHUNKING_ENABLED)

    drafts_dir = repo_root / "eval" / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = drafts_dir / f"kinegraph_benchmark_{output_version}.csv"
    audit_path = drafts_dir / f"kinegraph_benchmark_{output_version}.audit.json"
    manifest_path = drafts_dir / f"kinegraph_benchmark_{output_version}.generation.json"
    note_path = repo_root / "reports" / "synthetic_testset_spike.md"

    accepted_csv = repo_root / "eval" / "kinegraph_benchmark_v1.csv"
    if csv_path.resolve() == accepted_csv.resolve() and not force_new_version:
        raise ValueError(
            "refusing to overwrite the frozen benchmark CSV; "
            "use a draft output_version or pass force_new_version=True"
        )

    documents = load_markdown_documents(docs_dir_path)
    if not documents:
        raise FileNotFoundError(f"No markdown documents found under {docs_dir_path}")

    records = chunk_documents_for_generation(
        documents,
        adaptive_enabled=adaptive_enabled,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not records:
        raise RuntimeError("chunking produced zero records")
    lc_chunks = records_to_langchain_docs(records)

    if dry_run_chunks_only:
        # Phase-0 offline path: emit schema-valid single/multi-hop rows grounded in chunks.
        import pandas as pd

        first = records[0].text
        second = records[min(1, len(records) - 1)].text
        offline_rows = [
            {
                "user_input": "What does the first retrieved source chunk describe?",
                "reference_contexts": repr([first]),
                "reference": first[:500],
                "persona_name": "Offline Spike Reviewer",
                "query_style": "SYNTHETIC",
                "query_length": "SHORT",
                "synthesizer_name": "single_hop_specific_query_synthesizer",
            },
            {
                "user_input": "How do the first two source chunks relate in the corpus?",
                "reference_contexts": repr([first, second]),
                "reference": f"{first[:250]} ... {second[:250]}",
                "persona_name": "Offline Spike Reviewer",
                "query_style": "SYNTHETIC",
                "query_length": "MEDIUM",
                "synthesizer_name": "multi_hop_specific_query_synthesizer",
            },
        ]
        frame = pd.DataFrame(offline_rows)[list(REQUIRED_COLUMNS)]
        frame.to_csv(csv_path, index=False)
        audit = build_draft_audit(
            csv_path,
            repo_root,
            dataset_version=output_version,
            id_prefix="KGSYN",
        )
        write_reference_audit(audit_path, audit)
        usage = {"cost_complete": False, "note": "dry_run_chunks_only"}
        synthesizer_counts = {
            "single_hop_specific_query_synthesizer": 1,
            "multi_hop_specific_query_synthesizer": 1,
        }
        manifest = build_generation_manifest(
            repo_root=repo_root,
            csv_path=csv_path,
            audit_path=audit_path,
            row_count=len(frame),
            adaptive_enabled=adaptive_enabled,
            chunk_policy_version=CHUNK_POLICY_VERSION if adaptive_enabled else "legacy.recursive",
            generation_model=generation_model,
            embedding_model=embedding_model,
            testset_size=size,
            elapsed_seconds=0.0,
            usage={**usage, "chunk_count": len(records)},
            docs_dir=str(docs_dir_path.relative_to(repo_root)),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = GenerationResult(
            csv_path=csv_path,
            audit_path=audit_path,
            manifest_path=manifest_path,
            note_path=note_path if write_spike_note_file else None,
            row_count=len(frame),
            elapsed_seconds=0.0,
            chunk_policy_version=manifest["chunk_policy_version"],
            adaptive_enabled=adaptive_enabled,
            usage=manifest["usage"],
        )
        if write_spike_note_file:
            write_spike_note(note_path, result=result, synthesizer_counts=synthesizer_counts)
        return result

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY or OPENAI_API_KEY is required for generation")

    from openai import OpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.testset import TestsetGenerator

    base_url = (
        os.getenv("RAGAS_JUDGE_BASE_URL")
        or (
            "https://openrouter.ai/api/v1"
            if api_key.startswith("sk-or-") or "openrouter" in api_key
            else None
        )
    )
    client = OpenAI(api_key=api_key, base_url=base_url)
    tracker = UsageTracker()
    ragas_llm = llm_factory(model=generation_model, client=client)
    ragas_embeddings = OpenAIEmbeddings(client=client, model=embedding_model)
    generator = TestsetGenerator(llm=ragas_llm, embedding_model=ragas_embeddings)
    query_distribution = _build_query_distribution(ragas_llm)

    started = time.perf_counter()
    testset = generator.generate_with_chunks(
        chunks=lc_chunks,
        testset_size=size,
        query_distribution=query_distribution,
        raise_exceptions=True,
    )
    elapsed = time.perf_counter() - started
    frame = _normalize_dataframe(testset.to_pandas())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    audit = build_draft_audit(
        csv_path,
        repo_root,
        dataset_version=output_version,
        id_prefix="KGSYN",
    )
    write_reference_audit(audit_path, audit)

    usage = tracker.snapshot()
    usage.setdefault("cost_complete", False)
    usage["elapsed_seconds"] = round(elapsed, 3)
    usage["chunk_count"] = len(records)

    manifest = build_generation_manifest(
        repo_root=repo_root,
        csv_path=csv_path,
        audit_path=audit_path,
        row_count=len(frame),
        adaptive_enabled=adaptive_enabled,
        chunk_policy_version=CHUNK_POLICY_VERSION if adaptive_enabled else "legacy.recursive",
        generation_model=generation_model,
        embedding_model=embedding_model,
        testset_size=size,
        elapsed_seconds=elapsed,
        usage=usage,
        docs_dir=str(docs_dir_path.relative_to(repo_root)),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    synthesizer_counts: dict[str, int] = {}
    for value in frame["synthesizer_name"].tolist():
        synthesizer_counts[str(value)] = synthesizer_counts.get(str(value), 0) + 1

    result = GenerationResult(
        csv_path=csv_path,
        audit_path=audit_path,
        manifest_path=manifest_path,
        note_path=note_path if write_spike_note_file else None,
        row_count=len(frame),
        elapsed_seconds=round(elapsed, 3),
        chunk_policy_version=manifest["chunk_policy_version"],
        adaptive_enabled=adaptive_enabled,
        usage=usage,
    )
    if write_spike_note_file:
        write_spike_note(note_path, result=result, synthesizer_counts=synthesizer_counts)
    return result


def build_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--output-version", default="v1.2.0-draft")
    parser.add_argument("--adaptive-chunking", action="store_true")
    parser.add_argument("--recursive-chunking", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--generation-model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--force-new-version", action="store_true")
    parser.add_argument("--dry-run-chunks-only", action="store_true")
    parser.add_argument("--no-spike-note", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_cli(argv)
    adaptive: bool | None
    if args.adaptive_chunking and args.recursive_chunking:
        raise SystemExit("choose only one of --adaptive-chunking or --recursive-chunking")
    if args.adaptive_chunking:
        adaptive = True
    elif args.recursive_chunking:
        adaptive = False
    else:
        adaptive = None
    repo_root = Path(__file__).resolve().parents[1]
    result = generate_testset(
        repo_root=repo_root,
        docs_dir=args.docs_dir,
        size=args.size,
        output_version=args.output_version,
        adaptive_enabled=adaptive,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        generation_model=args.generation_model,
        embedding_model=args.embedding_model,
        force_new_version=args.force_new_version,
        write_spike_note_file=not args.no_spike_note,
        dry_run_chunks_only=args.dry_run_chunks_only,
    )
    print(f"Wrote draft CSV: {result.csv_path}")
    print(f"Wrote draft audit: {result.audit_path}")
    print(f"Wrote generation manifest: {result.manifest_path}")
    if result.note_path:
        print(f"Wrote spike note: {result.note_path}")
    print(
        f"rows={result.row_count} elapsed_s={result.elapsed_seconds} "
        f"policy={result.chunk_policy_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
