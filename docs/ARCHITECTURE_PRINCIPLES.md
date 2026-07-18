# Kinegraph Architecture Principles

## Vision

Kinegraph will be an evidence-first knowledge retrieval system that combines semantic search and explicit graph reasoning to produce answers that are trustworthy, explainable, and operationally verifiable.

The system should help a user understand not only **what** the answer is, but also:

- Which source chunks support it
- Which entities and relationships contributed to it
- How the retrieval workflow reached that evidence
- Where evidence is incomplete, conflicting, or uncertain

Kinegraph should favor transparent reasoning over opaque sophistication and grounded answers over plausible-looking completeness.

## Goal

Kinegraph's primary goal is to retrieve the smallest sufficient set of verified evidence for a question, explain how that evidence was found, and generate only what the evidence supports.

The architecture should continuously improve these measurable outcomes:

| Outcome | Target |
|---|---:|
| Faithfulness | ≥ 0.75 |
| Answer relevancy | ≥ 0.65 |
| Context recall | ≥ 0.65 |
| Answer correctness | ≥ 0.60 |
| Context precision | ≥ 0.90 |

These metrics are constraints on the architecture, not reasons to optimize one retrieval technique in isolation. An improvement is successful only when it strengthens overall answer quality without introducing unacceptable regressions in grounding, precision, latency, cost, or explainability.

## Architectural North Star

> Retrieve verified evidence first. Explain how it was found. Generate only what the evidence supports.

## Principles

### 1. Evidence is the source of truth

- Every answer claim should be traceable to retrieved source content.
- Graph entities must link to verified source chunks.
- Chunk identifiers must resolve to real vector-store records.
- Relationship evidence must remain attached to graph paths.
- Generated or hypothetical content must never be presented as evidence.
- When evidence is unavailable, the system should acknowledge uncertainty.

### 2. Retrieval comes before generation

The preferred workflow is:

```text
Understand query → retrieve → traverse → fuse → rerank → generate from evidence
```

Generation should synthesize retrieved knowledge. It should not compensate for missing or irrelevant context.

### 3. Use the simplest effective retrieval path

- Use vector retrieval for semantic and conceptual questions.
- Use graph retrieval for entities, dependencies, and relationships.
- Use hybrid retrieval when both signals provide value.
- Use vectorless retrieval for direct-document and lexical use cases.
- Activate additional retrieval mechanisms only when observable weakness justifies them.

### 4. Recovery mechanisms are conditional

Query expansion, decomposition, or a future HyDE experiment should run only when normal retrieval is weak.

Weakness signals can include:

- No graph seed
- Too few retrieved chunks
- Low vector similarity
- Low reranker confidence
- Insufficient source diversity
- No evidence-bearing relationship path
- Poor coverage of a compound question

Recovery occurs before RRF so all valid candidates enter the same fusion process.

### 5. The original query remains authoritative

- Preserve the original query throughout the workflow.
- Score and rerank results against the original question.
- Treat rewritten queries as recall aids, not replacements for user intent.
- Do not extract authoritative graph seeds directly from hypothetical text.
- Record which transformation produced each candidate.

### 6. Graph traversal is bounded and explainable

- Enforce a configurable hop limit.
- Prevent cycles.
- Return complete relationship paths.
- Report traversal strategy and depth.
- Preserve direction, weight, and evidence for every edge.
- Prefer community-constrained traversal when global expansion creates noise.

Every graph result should be able to answer: “How did the system reach this node?”

### 7. Vector and graph stores have distinct responsibilities

- ChromaDB owns embeddings and semantic retrieval.
- Neo4j owns entities, relationships, topology, and traversal.
- Stable, verified identifiers connect the stores.
- Embeddings should not be duplicated into Neo4j without a measured reason.

### 8. Fusion combines evidence; it does not validate it

- Only valid candidates enter RRF.
- Preserve source-specific and pre-fusion scores.
- Track contributing retrieval channels.
- Deduplicate using stable identifiers where possible.
- Apply relevance filtering after fusion.
- Agreement between weak channels is not proof of correctness.

### 9. Reranking uses the user's actual question

Textual relevance to the original query remains the primary signal. Relationship weight, traversal depth, centrality, evidence availability, and cross-channel agreement may act as secondary signals, but graph importance must not override poor semantic relevance.

### 10. Faithfulness takes priority over apparent completeness

When context is incomplete:

- Answer only the supported portion.
- Identify missing evidence.
- Lower confidence.
- Avoid filling gaps with model knowledge.
- Do not manufacture relationships to create a coherent narrative.

### 11. Confidence is based on observable signals

Confidence should incorporate evidence coverage, retrieval relevance, cross-store verification, path completeness, channel agreement, conflicts, and missing context. It should not be an unconstrained LLM self-rating.

### 12. Advanced features begin as experiments

Before becoming a default, a feature must demonstrate:

- Improvement on a fixed, versioned benchmark
- No unacceptable precision regression
- Acceptable latency and cost
- Stable behavior across query categories
- Clear observability and rollback
- Tests for expected and failure behavior

This applies to HyDE, decomposition, graph-aware reranking, new embedding models, and deeper traversal.

### 13. Benchmarks are architecture constraints

Benchmark runs must record the code revision, dataset version, retrieval configuration, generation and evaluation models, evaluation mode, per-query retrieval paths, latency, and token cost. Aggregate improvements must not hide regressions within individual query categories.

### 14. Observability is part of the retrieval contract

Each query should expose enough information to reconstruct the decision:

- Original and rewritten query
- Classified intent and retrieval mode
- Recovery trigger
- Candidate source and score
- RRF contribution and reranker score
- Graph seed, path, strategy, and depth
- Supporting chunk identifiers
- Stage-level latency

### 15. Failure is visible

- Missing Chroma link marks enrichment incomplete.
- Graph traversal failure retains base results and records the failure.
- RAGAS failure marks the run invalid or identifies heuristic fallback.
- Cross-encoder failure discloses keyword fallback.
- Empty retrieval returns an evidence-unavailable response.
- Infrastructure outages identify the unavailable retrieval channel.

### 16. Ingestion establishes the complete retrieval contract

A successful ingestion means chunks are embedded, entities and relationships are created, graph-to-vector links are verified, relationship evidence and weights are present, positioning metadata is computed, and automatic enrichment completed. The operation must return a validation summary.

### 17. Complexity requires measured justification

A component earns its place by improving retrieval quality, faithfulness, latency, cost, operability, explainability, or reliability. Features that add complexity without measured benefit remain experimental or are removed.

### 18. Prefer reversible decisions

Retrieval strategies should be configurable, observable, independently testable, and easy to disable. Data migrations should be additive and idempotent whenever practical.

## Decision Gate for New Features

Before implementation, answer:

1. Which measured failure does this address?
2. Which query categories are affected?
3. Can existing mechanisms solve the problem?
4. What metric improvement is expected?
5. What precision, latency, or cost regression is acceptable?
6. What observable condition triggers the feature?
7. Which evidence may it consume?
8. Can it introduce unsupported entities or claims?
9. How will its output be traced?
10. What benchmark result allows it to become a default?
11. How can it be disabled or rolled back?

If these questions do not have concrete answers, the feature is not ready for implementation.
