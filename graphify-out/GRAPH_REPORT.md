# Graph Report - /Users/sendils/work/repo/kinetic-v/kinegraph-v  (2026-08-05)

## Corpus Check
- 98 files · ~61,351 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1022 nodes · 1971 edges · 62 communities (59 shown, 3 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 100 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Context Ranker
- Node Enrichment
- Query Recovery
- LiteParse Client
- Architecture ADRs
- Hybrid RAG Workflow
- Grounding Citations
- RAGAS Evaluator Tests
- Multi-Hop Retriever
- Query API Routes
- Schema Extractors
- Verification Framework
- Neo4j Graph Service
- Enrichment v2
- Document Ingest API
- Intent Classifier
- PropertyGraph Ingest
- LangGraph Retriever Node
- Multi-Hop Retriever v2
- Adaptive Chunking
- Idempotent Ingester
- Settings and Security
- Chroma Vector Service
- Dual-Store Architecture
- Ingestion Validation
- Composed Retriever
- Provenance Recording
- Fusion and Rerank
- Vectorless BM25
- Celery Document Tasks
- Entity Dedup Resolver
- Experiment Validation
- Embedding Wrapper
- Cypher Safety Checks
- Hybrid RAG Concepts
- Adaptive Routing Plans
- Adaptive Chunking Tests
- Entity Resolution Extractor
- Document Processor Utils
- Ingestion Pipeline Docs
- Benchmark Audit Policy
- Provenance Tests
- Health Check Routes
- Retrieval Failure Modes
- Store Separation Docs
- Benchmark Audit Tests
- Synthetic Eval Stack
- Config Validators
- Lazy Settings Proxy
- Multi-Hop Improvements
- Retrieval Acceptance
- Quickstart Embedding Docs
- Docker Health Stack
- Benchmark Row Loader
- Backend Package Root

## God Nodes (most connected - your core abstractions)
1. `HybridRAGWorkflow` - 41 edges
2. `ChromaService` - 32 edges
3. `WorkflowState` - 28 edges
4. `QueryRecoveryEngine` - 27 edges
5. `TraversalStrategy` - 27 edges
6. `Neo4jService` - 25 edges
7. `ContextRanker` - 24 edges
8. `IdempotentGraphIngester` - 22 edges
9. `QueryMode` - 19 edges
10. `NodeEnricher` - 19 edges

## Surprising Connections (you probably didn't know these)
- `test_intent_router_offloads_local_document_lookup()` --indirect_call--> `_load_vectorless_document()`  [INFERRED]
  tests/test_warning_hardening.py → backend/core/langgraph_workflow.py
- `FakeTraversal` --uses--> `QueryRequest`  [INFERRED]
  tests/test_multi_hop.py → backend/app/models.py
- `test_query_model_validates_active_fusion_weights()` --calls--> `QueryRequest`  [EXTRACTED]
  tests/test_multi_hop.py → backend/app/models.py
- `FakeTraversal` --uses--> `TraversalStrategy`  [INFERRED]
  tests/test_multi_hop.py → backend/graph_retrieval/multi_hop.py
- `test_safe_pdf_filename_is_preserved_only_for_display()` --calls--> `_validate_upload_filename()`  [EXTRACTED]
  tests/test_warning_hardening.py → backend/app/api/routes/ingest.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Hybrid Retrieval Fusion Pipeline** — docs_architecture_vector_agent, docs_architecture_graph_agent, docs_architecture_vectorless_agent, docs_architecture_fusion_node, docs_api_reciprocal_rank_fusion [EXTRACTED 1.00]
- **Tri-Branch Document Ingestion** — docs_architecture_ingestion_pipeline, docs_architecture_chromadb_service, docs_architecture_neo4j_service, docs_architecture_bm25_lexical_cache, docs_architecture_celery_workers [EXTRACTED 1.00]
- **Controlled Benchmark Experiment Loop** — docs_experiment_validation_ratchet_policy, docs_experiment_validation_acceptance_contract, docs_experiment_validation_weighted_objective, docs_experiment_validation_manifest_artifacts, docs_benchmark_reference_audit_audit_sidecar [EXTRACTED 1.00]
- **Kinetic-V 2.0 Evidence Pipeline** — docs_architecture_adr_001_adaptive_routing_adaptiverouting, docs_architecture_adr_003_retrieval_orchestration_retrievalorchestration, docs_architecture_adr_004_verification_framework_verificationframework [EXTRACTED 1.00]
- **PropertyGraphIndex Dual-Store Flow** — docs_superpowers_plans_2026_07_03_propertygraphindex_propertygraphindex, docs_superpowers_specs_2026_07_03_propertygraphindex_design_dualstore, docs_superpowers_plans_2026_07_03_propertygraphindex_kg_nodes [EXTRACTED 1.00]
- **Multi-Hop Traversal Improvements** — docs_improvements_validation_report_enhanced_chunk_context, docs_improvements_validation_report_multihop_scoring_v2, docs_improvements_validation_report_query_relevance_filtering [EXTRACTED 1.00]

## Communities (62 total, 3 thin omitted)

### Community 0 - "Context Ranker"
Cohesion: 0.06
Nodes (57): ContextRanker, _keyword_score(), Any, Counter, Context Ranker — KineticGraph-Vectra Semantic-first, graph-aware reranker…, Return (chunk, score) pairs sorted descending by relevance., Rerank *chunks* by relevance to *query* and return the top-k. Args: query: The…, Rerank once and emit deterministic survival reasons for ADR-003. (+49 more)

### Community 1 - "Node Enrichment"
Cohesion: 0.08
Nodes (27): ChromaChunkValidator, ContextChunkLink, KineticVNode, NodeEnricher, Any, Context-first graph-node enrichment with verified vector provenance (Enhanced…, Enrich Neo4j entities using graph evidence verified against Chroma.…, Compute relevance weights for each chunk based on multiple factors. (+19 more)

### Community 2 - "Query Recovery"
Cohesion: 0.10
Nodes (27): FacetCoverage, Any, QueryRecoveryEngine, Conditional query decomposition, vocabulary expansion, and constrained HyDE., Plan recovery only after ordinary retrieval produces weak candidates., Measure whether each literal question facet appears in any candidate., RecoveryPlan, WeaknessAssessment (+19 more)

### Community 3 - "LiteParse Client"
Cohesion: 0.07
Nodes (24): LiteParseClient, LiteParseUnavailableError, Any, LiteParse HTTP client. Streams local documents to the self-hosted LiteParse…, Raised when the LiteParse container cannot be reached or times out., Thin HTTP wrapper around the local LiteParse parsing service., Perform a lightweight liveness probe. The versioned API endpoint is probed with…, Stream a local document to LiteParse and return structured Markdown. Args:… (+16 more)

### Community 4 - "Architecture ADRs"
Cohesion: 0.06
Nodes (39): Adaptive Routing, ADR-001: Adaptive Routing, enable_adaptive_routing, kinegraph.adaptive-routing.v1, ADAPTIVE_CHUNKING_ENABLED, Adaptive Chunking, ADR-002: Adaptive Chunking, IdempotentGraphIngester (+31 more)

### Community 5 - "Hybrid RAG Workflow"
Cohesion: 0.10
Nodes (23): DocumentChunk, Document chunk result, HybridRAGWorkflow, Any, TraversalStrategy, Convert reranked chunks to DocumentChunk objects., Expose raw internal evidence for evaluator-side redaction/persistence., Execute the full retrieval + generation workflow. (+15 more)

### Community 6 - "Grounding Citations"
Cohesion: 0.11
Nodes (25): apply_critic_response(), assign_citation_ids(), build_citation_context(), format_grounded_claims(), _parse_json_object(), Any, Deterministic citation contracts for grounded answer generation., Allow the critic only to retain existing claims; it cannot add or rewrite. (+17 more)

### Community 7 - "RAGAS Evaluator Tests"
Cohesion: 0.10
Nodes (24): _make_provenance(), Build a results DataFrame with realistic provenance for diagnostic tests., empty_seed, traversal_failure, cycle_prevention, missing_evidence are summed…, p50 and p95 percentiles are both present and ordered correctly., pre_fusion, reranking, and final_truncation drops are counted independently., all_complete is False when complete_path_count < traversal_candidate_count., p50 and p95 are None when no provenance rows carry graph latency data., sample_ids gracefully falls back to [] when sample_id column is absent. (+16 more)

### Community 8 - "Multi-Hop Retriever"
Cohesion: 0.13
Nodes (16): MultiHopGraphRetriever, Any, Compute adaptive depth penalty based on query complexity. Formula: 0.2 * (depth…, Enhanced multi-hop graph traversal with improved scoring and relevance. Key…, Extract meaningful terms from query (excluding stopwords)., Compute semantic relevance between query and node based on keyword overlap.…, MultiHopGraphRetriever, _edge() (+8 more)

### Community 9 - "Query API Routes"
Cohesion: 0.11
Nodes (22): get, post, Request, query_system(), Query Endpoints for Hybrid RAG — v2 Uses execute_with_answer() to surface…, Test endpoint to verify the query system is operational., Query the hybrid RAG system. v2 improvements: - Intent classification routes…, test_query() (+14 more)

### Community 10 - "Schema Extractors"
Cohesion: 0.14
Nodes (19): get_extractor_stack(), Construct the stacked extractor pipeline: 1. SchemaLLMPathExtractor: Strict,…, Subclass of SimpleLLMPathExtractor that tags all extracted relationships with…, TaggedSimpleLLMPathExtractor, audit_schema_coverage(), load_graph_snapshot(), Any, Read-only ontology coverage analysis for a persisted graph and golden benchmark. (+11 more)

### Community 11 - "Verification Framework"
Cohesion: 0.18
Nodes (23): Apply ADR-004 only when explicitly enabled; scoring never overrides refusal., apply_response_policy(), _bounded(), build_verification_outcome(), calibrate_kinetic_score(), compute_kinetic_score(), _explicit_conflicts(), _metadata_link_consistency() (+15 more)

### Community 12 - "Neo4j Graph Service"
Cohesion: 0.09
Nodes (15): GraphWriteResult, Neo4jService, Any, Service for interacting with Neo4j Graph Database, Initialize Neo4j driver, Close the Neo4j driver, Verify connection to Neo4j, Create indexes for better query performance (+7 more)

### Community 13 - "Enrichment v2"
Cohesion: 0.16
Nodes (13): ChromaChunkValidator, ContextChunkLink, KineticVNode, NodeEnricher, Any, Context-first graph-node enrichment with verified vector provenance (Enhanced…, Enrich Neo4j entities using graph evidence verified against Chroma.…, Compute relevance weights for each chunk based on multiple factors. (+5 more)

### Community 14 - "Document Ingest API"
Cohesion: 0.12
Nodes (17): _allocate_upload_path(), ingest_document(), Path, post, Return a display-safe basename or reject path-like upload names., Allocate a server-controlled filename beneath the configured directory., Ingest a document (PDF) and process it asynchronously The document will be: 1.…, _validate_upload_filename() (+9 more)

### Community 15 - "Intent Classifier"
Cohesion: 0.16
Nodes (20): analyze_query_signals(), classify_intent(), extract_query_facets(), Any, Query Intent Classifier — KineticGraph-Vectra Classifies a query into intent…, Extract observable routing signals without generating graph seeds., Match whole trigger phrases instead of arbitrary substrings., Return explicit question/request facets without generating new intent. (+12 more)

### Community 16 - "PropertyGraph Ingest"
Cohesion: 0.15
Nodes (15): Idempotent PropertyGraph ingestion with ADR-002 chunk contracts., CustomNeo4jPropertyGraphStore, get_chroma_vector_store(), get_llm(), get_neo4j_graph_store(), Subclass of Neo4jPropertyGraphStore that overrides upsert_nodes to fix Cypher…, Get the Neo4j graph store instance., Get the ChromaDB vector store instance, pointing to the dedicated `kg_nodes`… (+7 more)

### Community 17 - "LangGraph Retriever Node"
Cohesion: 0.16
Nodes (14): LangGraphGraphRetrieverNode, Any, TraversalStrategy, Retrieves matching chunks and returns standard formatted dicts for LangGraph…, Return graph candidates and traversal diagnostics without shared mutable state., Wraps the ComposedGraphRetriever to match the interface contract expected by…, Enum, str (+6 more)

### Community 18 - "Multi-Hop Retriever v2"
Cohesion: 0.18
Nodes (10): MultiHopGraphRetriever, Any, Enum, str, Enhanced multi-hop traversal with query-relevance scoring and semantic…, Compute adaptive depth penalty based on query complexity. Formula: 0.2 * (depth…, Enhanced multi-hop graph traversal with improved scoring and relevance. Key…, Extract meaningful terms from query (excluding stopwords). (+2 more)

### Community 19 - "Adaptive Chunking"
Cohesion: 0.16
Nodes (16): _adaptive_chunk(), _chunk_recursive_records(), ChunkRecord, content_hash(), _is_table_row(), _make_record(), _parse_blocks(), Any (+8 more)

### Community 20 - "Idempotent Ingester"
Cohesion: 0.16
Nodes (13): SHA-256-derived stable chunk ID scoped to document + policy., stable_chunk_id(), IdempotentGraphIngester, Split text into overlapping chunks using the recursive fallback policy., Build ADR-002 chunk records (adaptive when enabled)., Alias for build_chunks (structural-first when adaptive flag is on)., Ingests files/folders into PropertyGraphIndex idempotently. Uses content…, Compute SHA-256 hash of a string. (+5 more)

### Community 21 - "Settings and Security"
Cohesion: 0.15
Nodes (12): Config, Settings, _persist_document(), Any, Run all async ingestion work in one event loop and close every client., BaseSettings, _task(), test_async_ingestion_closes_chroma_and_neo4j_clients() (+4 more)

### Community 22 - "Chroma Vector Service"
Cohesion: 0.13
Nodes (11): ChromaService, Any, ChromaDB Service for Vector Storage, Delete the collection (useful for testing), Get the number of documents in the collection, Service for interacting with ChromaDB, Initialize ChromaDB client, Release the shared Chroma HTTP client and its connection pool. (+3 more)

### Community 23 - "Dual-Store Architecture"
Cohesion: 0.12
Nodes (19): ChromaDB, Multi-Index Architecture, Neo4j, ComposedGraphRetriever, EntityResolver, kg_nodes (Chroma collection), OntologySchema, PropertyGraphIndex Dual-Store Integration Plan (+11 more)

### Community 24 - "Ingestion Validation"
Cohesion: 0.18
Nodes (12): build_ingestion_validation_report(), Additive, idempotent validation summary — never invents missing links., Any, Check a document-scoped chunk hash in the graph database., Filter already-ingested chunks and build LlamaIndex nodes with stable IDs., Ingests a single file (PDF or Markdown/text) into the PropertyGraphIndex.…, Ingests all supported files from a directory in a single batch to avoid event…, Enrich only entities touched by this ingestion and verify Chroma links. (+4 more)

### Community 25 - "Composed Retriever"
Cohesion: 0.17
Nodes (11): Health Check Endpoints, get_task_status(), get, Document Ingestion Endpoints, Check the status of a document processing task, get, FastAPI Application Entry Point — KineticGraph-Vectra Includes observability…, root() (+3 more)

### Community 26 - "Provenance Recording"
Cohesion: 0.19
Nodes (9): BM25Retriever, Any, Save chunk structures to a JSON cache and raw text to a text file., Process and query an attachment. If it's small, returns the whole thing. If…, Standard Okapi BM25 implementation. Reference:…, Load cached JSON chunk files, filter them, and run BM25 search., Args: corpus: List of chunk dicts. Each must have "content" and "metadata". k1:…, Simple tokenizer to split text into words, removing punctuation. (+1 more)

### Community 27 - "Fusion and Rerank"
Cohesion: 0.20
Nodes (15): _manifest(), p95 latency > 125% of baseline must block promotion., Non-zero traversal_failure_count must block promotion., Changing more than max_hops must block promotion (one-lever rule)., Category recall improvement below 0.05 must block promotion., Overall context_precision below 0.90 must block promotion even if recall…, test_sweep_preserves_baseline_as_rollback_default(), test_sweep_rejects_missing_hop_manifest() (+7 more)

### Community 28 - "Vectorless BM25"
Cohesion: 0.13
Nodes (11): Expand / rewrite the query to improve recall. Adds intent-specific context…, rewrite_query_for_retrieval(), _load_vectorless_document(), Classify intent, rewrite query for retrieval, and set mode., Run local document lookup without blocking the async workflow., Run attachment chunking and lookup outside the event loop., _search_vectorless_attachment(), Vectorless RAG Service — KineticGraph-Vectra Implements a pure-Python BM25… (+3 more)

### Community 29 - "Celery Document Tasks"
Cohesion: 0.16
Nodes (12): build_document_chunks(), Build versioned ChunkRecords via the ADR-002 contract., CallbackTask, health_check(), process_document(), Celery Tasks for Document Processing, Process a document: extract text, chunk, embed, extract entities Args:…, Simple health check task for monitoring worker status (+4 more)

### Community 30 - "Entity Dedup Resolver"
Cohesion: 0.22
Nodes (9): cosine_similarity(), EntityResolver, Any, Deduplicates entity names using exact-match and semantic embedding similarity…, Deduplicates a list of entity names against each other and against cached…, Compute cosine similarity between two vectors., FakeEmbedModel, test_cosine_similarity() (+1 more)

### Community 31 - "Experiment Validation"
Cohesion: 0.26
Nodes (9): _manifest(), parametrize, _report(), test_manifest_records_dataset_hash_revision_config_and_models(), test_manifest_uses_effective_reference_audit_identity(), test_metric_validation_rejects_invalid_values(), test_ratchet_keeps_one_lever_improvement(), test_ratchet_rejects_confounded_or_multi_lever_experiments() (+1 more)

### Community 32 - "Embedding Wrapper"
Cohesion: 0.20
Nodes (4): LangChainEmbeddingWrapper, Any, BaseEmbedding, test_embedding_wrapper()

### Community 33 - "Cypher Safety Checks"
Cohesion: 0.20
Nodes (11): _mask_cypher_literals(), Neo4j Service for Graph Storage, Raised when generated Cypher exceeds the read-only query contract., Replace Cypher literals with spaces without regex backtracking. Generated…, Validate an LLM query and add the only permitted result bound., UnsafeCypherError, validate_read_only_cypher(), parametrize (+3 more)

### Community 34 - "Hybrid RAG Concepts"
Cohesion: 0.20
Nodes (12): DocumentChunk Schema, Grounding Critique, Hybrid RAG Query Endpoint, Query Modes (vector, graph, hybrid), FastAPI API Layer, KineticGraph-Vectra Hybrid RAG, Evidence-First Knowledge Retrieval, Faithfulness Over Completeness (+4 more)

### Community 35 - "Adaptive Routing Plans"
Cohesion: 0.29
Nodes (8): _alternatives(), build_execution_plan(), _channels(), ExecutionPlan, Any, Deterministic, provenance-first execution planning for ADR-001., Auditable retrieval plan; it never generates evidence or an answer., Build an execution plan while preserving explicit modes and weak signals.

### Community 36 - "Adaptive Chunking Tests"
Cohesion: 0.33
Nodes (9): chunk_document(), Produce versioned chunks. Default (adaptive_enabled=False) preserves recursive-…, ADR-002 adaptive chunking contract tests., test_adaptive_policy_emits_structural_table_and_image_chunks(), test_chunk_ids_differ_across_documents_with_identical_text(), test_chunk_metadata_is_chroma_flat_and_includes_provenance(), test_disabled_policy_uses_recursive_chunks_with_stable_ids(), test_oversized_section_falls_back_to_recursive_with_parent_link() (+1 more)

### Community 37 - "Entity Resolution Extractor"
Cohesion: 0.36
Nodes (5): EntityResolutionExtractor, Any, TransformComponent that resolves and deduplicates entities across all nodes in-…, BaseNode, TransformComponent

### Community 38 - "Document Processor Utils"
Cohesion: 0.20
Nodes (9): chunk_text(), extract_entities_and_relationships(), generate_chunk_id(), Any, Document Processing Utilities, Extract entities and relationships from text using LLM Args: text: Text to…, Generate a unique ID for a chunk Args: content: Chunk content index: Chunk…, Split text into chunks. Legacy recursive splitter used when adaptive chunking… (+1 more)

### Community 39 - "Ingestion Pipeline Docs"
Cohesion: 0.20
Nodes (10): BM25 Local Lexical Cache, Celery Worker Layer, Document Ingestion Pipeline, Vectorless Agent (BM25), Evaluation Mode Profiles, Graph Construction Layer, LiteParseClient Driver, LiteParse Server Integration (+2 more)

### Community 40 - "Benchmark Audit Policy"
Cohesion: 0.22
Nodes (10): Decision Gate for New Features, RAGAS Metric Targets, audit_benchmark_references.py, Benchmark Audit Sidecar v1.1.0-draft, Benchmark Human Review Workflow, Benchmark Acceptance Contract, RAGAS Manifest Artifacts, OpenResearch-Inspired Ratchet Policy (+2 more)

### Community 41 - "Provenance Tests"
Cohesion: 0.44
Nodes (8): parametrize, _record(), _result(), test_graph_paths_and_retriever_behavior_are_audited(), test_jsonl_is_deterministic_and_diagnostics_group_failures(), test_offline_failure_fixtures_are_schema_valid(), test_profile_escape_is_recorded_and_grouped_as_failure(), test_provenance_is_versioned_reconstructable_and_redacts_secrets()

### Community 42 - "Health Check Routes"
Cohesion: 0.29
Nodes (8): health_check(), liveness(), get, Request, Health check endpoint that verifies all services are operational, Kubernetes liveness probe, Kubernetes readiness probe, readiness()

### Community 43 - "Retrieval Failure Modes"
Cohesion: 0.25
Nodes (8): Reciprocal Rank Fusion, Fusion Node (RRF), Conditional Recovery Mechanisms, Context Recall Coverage Problem, Hybrid Retrieval Failure Path, Intent Classification and Routing, Silent Mode Downgrade, Stale Score Provenance

### Community 44 - "Store Separation Docs"
Cohesion: 0.29
Nodes (8): ChromaDB Service, Graph Agent, LangGraph Orchestration Layer, Neo4j Service, Vector and Graph Store Separation, Vector Agent, Generated Cypher Validation, Neo4j Credential Policy

### Community 45 - "Benchmark Audit Tests"
Cohesion: 0.61
Nodes (7): _accepted_audit(), _rehash(), _rows(), test_changed_reference_without_human_review_is_rejected(), test_checked_in_audit_covers_all_rows_and_is_accepted_for_evaluation(), test_fully_reviewed_versioned_audit_is_accepted_and_filters_rows(), test_schema_rejects_missing_stable_id_and_stale_dataset_hash()

### Community 47 - "Config Validators"
Cohesion: 0.33
Nodes (3): model_validator, model_validator, ValueError

### Community 48 - "Lazy Settings Proxy"
Cohesion: 0.33
Nodes (5): get_settings(), _LazySettings, Any, Build settings on first use so imports remain side-effect free., Backward-compatible lazy proxy for modules that import ``settings``.

### Community 49 - "Multi-Hop Improvements"
Cohesion: 0.33
Nodes (7): Enhanced Chunk Context in Descriptions, enrichment_v2.py, multi_hop_v2.py, Enhanced Multi-Hop Scoring Formula, Query Relevance Filtering at Each Hop, RAGAS Benchmark Targets, Multi-Hop Traversal Improvements Validation Report

### Community 51 - "Quickstart Embedding Docs"
Cohesion: 0.50
Nodes (4): Document Ingestion Endpoint, OpenAI Embedding API Key Requirement, text-embedding-ada-002, Docker Compose Quick Start

### Community 52 - "Docker Health Stack"
Cohesion: 0.50
Nodes (4): Health Check Endpoints, Docker Compose Service Stack, Neo4j HTTP Healthcheck Fix, Ready Service Stack

### Community 54 - "Benchmark Row Loader"
Cohesion: 0.67
Nodes (3): load_benchmark_rows(), Path, Load question/reference text without assuming nonexistent claim/category fields.

## Ambiguous Edges - Review These
- `kg_nodes (Chroma collection)` → `kinetic_vectors (Chroma collection)`  [AMBIGUOUS]
  docs/v3_schema_audit.md · relation: semantically_similar_to

## Knowledge Gaps
- **40 isolated node(s):** `Config`, `Health Check Endpoints`, `DocumentChunk Schema`, `Celery Worker Layer`, `RAGAS Metric Targets` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `kg_nodes (Chroma collection)` and `kinetic_vectors (Chroma collection)`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **Why does `ChromaService` connect `Chroma Vector Service` to `Cypher Safety Checks`, `Hybrid RAG Workflow`, `Grounding Citations`, `Query API Routes`, `Neo4j Graph Service`, `Document Ingest API`, `PropertyGraph Ingest`, `LangGraph Retriever Node`, `Idempotent Ingester`, `Settings and Security`, `Composed Retriever`, `Celery Document Tasks`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Why does `HybridRAGWorkflow` connect `Hybrid RAG Workflow` to `Context Ranker`, `Query Recovery`, `Grounding Citations`, `Query API Routes`, `Verification Framework`, `Neo4j Graph Service`, `Document Ingest API`, `Intent Classifier`, `LangGraph Retriever Node`, `Chroma Vector Service`, `Vectorless BM25`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `ContextRanker` connect `Context Ranker` to `Hybrid RAG Workflow`, `Grounding Citations`, `Config Validators`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `HybridRAGWorkflow` (e.g. with `DocumentChunk` and `QueryMode`) actually correct?**
  _`HybridRAGWorkflow` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `ChromaService` (e.g. with `HybridRAGWorkflow` and `WorkflowState`) actually correct?**
  _`ChromaService` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `WorkflowState` (e.g. with `DocumentChunk` and `QueryMode`) actually correct?**
  _`WorkflowState` has 9 INFERRED edges - model-reasoned connections that need verification._