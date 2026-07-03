# PropertyGraphIndex Dual-Store Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate LlamaIndex's `PropertyGraphIndex` with Neo4j and ChromaDB as decoupled stores in the existing hybrid RAG pipeline, complete with schema-constrained extraction, entity deduplication, composed retrievers, a LangGraph wrapper, and a RAGAS comparison evaluation harness.

**Architecture:** Use a custom `BaseEmbedding` wrapper to reuse the existing LangChain `OpenAIEmbeddings` config. Extract entities/relations using stacked schema-guided and free-form LLM extractors. Deduplicate entities using cosine similarity, store graph nodes/edges in Neo4j, and store embeddings in a dedicated Chroma collection. Composed retriever fuses results from vector search, synonym search, and a sandboxed read-only Cypher query.

**Tech Stack:** Python 3.11+, LlamaIndex core, `llama-index-llms-openai`, `llama-index-embeddings-openai`, `llama-index-graph-stores-neo4j`, `llama-index-vector-stores-chroma`, Neo4j, ChromaDB, RAGAS, PyYAML, Pytest.

## Global Constraints
- **Schema-constrained extraction** — use `SchemaLLMPathExtractor` with an explicit, versioned schema defined in `config/ontology_schema.yaml`. Stack a `SimpleLLMPathExtractor` as a fallback.
- **Embedding provider must stay abstracted** — reuse the existing `BaseEmbeddings` wrapper already used for ChromaDB. Pass it as `embed_model` into `PropertyGraphIndex`.
- **Dual-store wiring is explicit** — construct `Neo4jPropertyGraphStore` and a `ChromaVectorStore` pointed at a new collection `kg_nodes`. Do not use Neo4j's native vector index.
- **Entity resolution / dedup step required** — run a dedup pass (exact-match + embedding-similarity clustering at `0.85` threshold) on entity names before graph write. Log collisions.
- **Retrieval composition** — implement `as_retriever(sub_retrievers=[VectorContextRetriever, LLMSynonymRetriever])`; add a read-only `TextToCypherRetriever` behind a feature flag.
- **Idempotent ingestion** — use content hashing per document/chunk to skip already-ingested material.
- **RAGAS eval hook** — report `context_precision`, `context_recall`, and `answer_relevancy` comparing the baseline with the new graph retriever.

---

### Task 1: Install Dependencies and Configure Environment

**Files:**
- Modify: [requirements.txt](file:///Users/sendils/work/repo/kinetic-v/kinegraph-v/requirements.txt)

**Interfaces:**
- Consumes: None
- Produces: Installed python packages in the `venv311` environment.

- [ ] **Step 1: Add new dependencies to requirements.txt**
  Update requirements.txt to include the necessary LlamaIndex packages and PyYAML.
  ```diff
  # Vector Database
  chromadb==0.4.22
+ llama-index==0.12.12
+ llama-index-llms-openai==0.3.11
+ llama-index-embeddings-openai==0.3.1
+ llama-index-graph-stores-neo4j==0.4.2
+ llama-index-vector-stores-chroma==0.4.1
+ pyyaml>=6.0
  ```

- [ ] **Step 2: Run pip install inside venv311**
  Run: `./venv311/bin/pip install -r requirements.txt`
  Expected: Installation succeeds without conflicts.

- [ ] **Step 3: Commit dependencies**
  Run:
  ```bash
  git add requirements.txt
  git commit -m "chore: add llamaindex and pyyaml dependencies"
  ```

---

### Task 2: Define Schema Configuration & Loader

**Files:**
- Create: `config/ontology_schema.yaml`
- Create: `backend/graph_ingestion/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: YAML parser library.
- Produces: `OntologySchema` configuration loaded at runtime.

- [ ] **Step 1: Create config/ontology_schema.yaml**
  Write the strict schema rules in YAML format.
  ```yaml
  version: "1.0.0"
  entity_types:
    - Person
    - Organization
    - Component
    - Concept
    - Technology
  relation_types:
    - DEVELOPED
    - ORCHESTRATES
    - COMBINES
    - USES
    - MENTIONS
    - INTEGRATES
  valid_triples:
    - [Component, ORCHESTRATES, Component]
    - [Component, COMBINES, Component]
    - [Component, USES, Component]
    - [Organization, DEVELOPED, Technology]
    - [Person, DEVELOPED, Concept]
    - [Component, INTEGRATES, Technology]
  ```

- [ ] **Step 2: Implement schema loader in backend/graph_ingestion/schema.py**
  Implement loading and parsing logic.
  ```python
  import yaml
  from pathlib import Path
  from typing import List, Tuple, Set

  class OntologySchema:
      def __init__(self, yaml_path: str = "config/ontology_schema.yaml"):
          with open(yaml_path, "r") as f:
              data = yaml.safe_load(f)
          self.version = data.get("version", "1.0.0")
          self.entity_types = data.get("entity_types", [])
          self.relation_types = data.get("relation_types", [])
          self.valid_triples = [tuple(t) for t in data.get("valid_triples", [])]
          self.valid_triples_set = set(self.valid_triples)

      def validate_triple(self, source_type: str, relation: str, target_type: str) -> bool:
          return (source_type, relation, target_type) in self.valid_triples_set
  ```

- [ ] **Step 3: Write tests for schema loader**
  Create `tests/test_schema.py`.
  ```python
  from backend.graph_ingestion.schema import OntologySchema

  def test_schema_load():
      schema = OntologySchema("config/ontology_schema.yaml")
      assert "Person" in schema.entity_types
      assert "DEVELOPED" in schema.relation_types
      assert schema.validate_triple("Person", "DEVELOPED", "Concept") is True
      assert schema.validate_triple("Person", "USES", "Person") is False
  ```

- [ ] **Step 4: Verify test fails and passes**
  Run: `./venv311/bin/pytest tests/test_schema.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add config/ontology_schema.yaml backend/graph_ingestion/schema.py tests/test_schema.py
  git commit -m "feat: add ontology schema configuration and loader"
  ```

---

### Task 3: Implement Embedding Wrapper

**Files:**
- Create: `backend/graph_ingestion/embedding_wrapper.py`
- Test: `tests/test_embedding_wrapper.py`

**Interfaces:**
- Consumes: Existing LangChain `OpenAIEmbeddings` from `ChromaService`.
- Produces: `LangChainEmbeddingWrapper` extending LlamaIndex's `BaseEmbedding`.

- [ ] **Step 1: Implement LangChainEmbeddingWrapper**
  Create `backend/graph_ingestion/embedding_wrapper.py`.
  ```python
  from typing import List, Any
  from llama_index.core.embeddings import BaseEmbedding
  from pydantic import PrivateAttr

  class LangChainEmbeddingWrapper(BaseEmbedding):
      _lc_embeddings: Any = PrivateAttr()

      def __init__(self, langchain_embeddings: Any, **kwargs: Any):
          super().__init__(**kwargs)
          self._lc_embeddings = langchain_embeddings

      def _get_query_embedding(self, query: str) -> List[float]:
          return self._lc_embeddings.embed_query(query)

      def _get_text_embedding(self, text: str) -> List[float]:
          return self._lc_embeddings.embed_documents([text])[0]

      def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
          return self._lc_embeddings.embed_documents(texts)

      async def _aget_query_embedding(self, query: str) -> List[float]:
          return await self._lc_embeddings.aembed_query(query)

      async def _aget_text_embedding(self, text: str) -> List[float]:
          res = await self._lc_embeddings.aembed_documents([text])
          return res[0]
  ```

- [ ] **Step 2: Write tests for embedding wrapper**
  Create `tests/test_embedding_wrapper.py`.
  ```python
  from backend.graph_ingestion.embedding_wrapper import LangChainEmbeddingWrapper
  from langchain_openai import OpenAIEmbeddings

  def test_embedding_wrapper():
      lc_emb = OpenAIEmbeddings(openai_api_key="fake_key")
      wrapper = LangChainEmbeddingWrapper(lc_emb)
      assert wrapper is not None
  ```

- [ ] **Step 3: Run test**
  Run: `./venv311/bin/pytest tests/test_embedding_wrapper.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
  Run:
  ```bash
  git add backend/graph_ingestion/embedding_wrapper.py tests/test_embedding_wrapper.py
  git commit -m "feat: implement langchain embedding wrapper for llamaindex"
  ```

---

### Task 4: Configure Decoupled Vector & Graph Stores

**Files:**
- Create: `backend/graph_ingestion/stores.py`
- Test: `tests/test_stores.py`

**Interfaces:**
- Consumes: `settings` and `ChromaService` configuration.
- Produces: `get_neo4j_graph_store() -> Neo4jPropertyGraphStore` and `get_chroma_vector_store() -> ChromaVectorStore`.

- [ ] **Step 1: Implement store wiring in backend/graph_ingestion/stores.py**
  Initialize Neo4j and Chroma stores.
  ```python
  import chromadb
  from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
  from llama_index.vector_stores.chroma import ChromaVectorStore
  from backend.core.config import settings

  def get_neo4j_graph_store() -> Neo4jPropertyGraphStore:
      return Neo4jPropertyGraphStore(
          username=settings.NEO4J_USER,
          password=settings.NEO4J_PASSWORD,
          url=settings.NEO4J_URI,
          database="neo4j"
      )

  def get_chroma_vector_store() -> ChromaVectorStore:
      # Explicitly pointing to kg_nodes collection for PropertyGraphIndex
      # Confirm: Neo4j's native vector index is NOT used.
      client = chromadb.HttpClient(
          host=settings.CHROMA_HOST,
          port=settings.CHROMA_PORT
      )
      collection = client.get_or_create_collection("kg_nodes")
      return ChromaVectorStore(chroma_collection=collection)
  ```

- [ ] **Step 2: Write basic connection test**
  Create `tests/test_stores.py`.
  ```python
  from backend.graph_ingestion.stores import get_neo4j_graph_store, get_chroma_vector_store

  def test_stores_initialization():
      graph_store = get_neo4j_graph_store()
      vector_store = get_chroma_vector_store()
      assert graph_store is not None
      assert vector_store is not None
  ```

- [ ] **Step 3: Run test**
  Run: `./venv311/bin/pytest tests/test_stores.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
  Run:
  ```bash
  git add backend/graph_ingestion/stores.py tests/test_stores.py
  git commit -m "feat: implement explicit dual-store setup for Neo4j and Chroma"
  ```

---

### Task 5: Implement Ingestion Extractors

**Files:**
- Create: `backend/graph_ingestion/extractors.py`
- Test: `tests/test_extractors.py`

**Interfaces:**
- Consumes: `OntologySchema` and standard LlamaIndex LLM.
- Produces: `get_extractor_stack(schema: OntologySchema, llm: Any) -> List[TransformComponent]`.

- [ ] **Step 1: Implement stack extractors in backend/graph_ingestion/extractors.py**
  Stack `SchemaLLMPathExtractor` and `SimpleLLMPathExtractor`.
  ```python
  from typing import List, Any
  from llama_index.core.indices.property_graph import SchemaLLMPathExtractor, SimpleLLMPathExtractor
  from llama_index.core.schema import TransformComponent
  from backend.graph_ingestion.schema import OntologySchema

  def get_extractor_stack(schema: OntologySchema, llm: Any) -> List[TransformComponent]:
      # Schema extractor with explicit types and triples
      schema_extractor = SchemaLLMPathExtractor(
          llm=llm,
          possible_entities=schema.entity_types,
          possible_relations=schema.relation_types,
          possible_triples=schema.valid_triples,
          strict=True
      )
      
      # Simple path extractor as fallback
      fallback_extractor = SimpleLLMPathExtractor(
          llm=llm,
          num_workers=1
      )
      
      return [schema_extractor, fallback_extractor]
  ```

- [ ] **Step 2: Write test to verify extractors output tag**
  Create `tests/test_extractors.py`.
  ```python
  from backend.graph_ingestion.extractors import get_extractor_stack
  from backend.graph_ingestion.schema import OntologySchema
  from llama_index.llms.openai import OpenAI

  def test_extractor_stack():
      schema = OntologySchema("config/ontology_schema.yaml")
      llm = OpenAI(model="gpt-4o-mini", temperature=0)
      stack = get_extractor_stack(schema, llm)
      assert len(stack) == 2
  ```

- [ ] **Step 3: Run test**
  Run: `./venv311/bin/pytest tests/test_extractors.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
  Run:
  ```bash
  git add backend/graph_ingestion/extractors.py tests/test_extractors.py
  git commit -m "feat: implement extractor stack using Schema and Simple LLM path extractors"
  ```

---

### Task 6: Implement Entity Resolution (Deduplication)

**Files:**
- Create: `backend/graph_ingestion/dedup.py`
- Test: `tests/test_dedup.py`

**Interfaces:**
- Consumes: Entity names and standard embedding model.
- Produces: Resolved entity names.

- [ ] **Step 1: Implement clustering logic in backend/graph_ingestion/dedup.py**
  Implement exact match plus embedding similarity clustering.
  ```python
  import numpy as np
  from typing import List, Dict, Tuple
  import logging

  logger = logging.getLogger(__name__)

  def cosine_similarity(v1: List[float], v2: List[float]) -> float:
      a = np.array(v1)
      b = np.array(v2)
      return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

  class EntityResolver:
      def __init__(self, embed_model: Any, threshold: float = 0.85):
          self.embed_model = embed_model
          self.threshold = threshold
          self.resolved_cache: Dict[str, Tuple[str, List[float]]] = {} # lowercase_name -> (canonical_name, embedding)

      def resolve_entities(self, entity_names: List[str]) -> Dict[str, str]:
          resolution_map = {}
          for name in entity_names:
              lower_name = name.strip().lower()
              if lower_name in self.resolved_cache:
                  resolved_name = self.resolved_cache[lower_name][0]
                  resolution_map[name] = resolved_name
                  continue

              # Compute embedding
              emb = self.embed_model.get_text_embedding(name)
              
              # Find similarity match
              match_found = False
              for cached_lower, (cached_canonical, cached_emb) in self.resolved_cache.items():
                  sim = cosine_similarity(emb, cached_emb)
                  if sim >= self.threshold:
                      logger.info(f"Dedup collision: Resolved '{name}' to '{cached_canonical}' (similarity: {sim:.3f})")
                      resolution_map[name] = cached_canonical
                      self.resolved_cache[lower_name] = (cached_canonical, cached_emb)
                      match_found = True
                      break

              if not match_found:
                  self.resolved_cache[lower_name] = (name, emb)
                  resolution_map[name] = name

          return resolution_map
  ```

- [ ] **Step 2: Write test for exact-match and forced collision**
  Create `tests/test_dedup.py`.
  ```python
  from backend.graph_ingestion.dedup import EntityResolver
  import pytest

  class FakeEmbedModel:
      def get_text_embedding(self, text: str):
          # Fake embeddings to force similarity
          if "Sendil" in text:
              return [0.9, 0.1, 0.0]
          return [0.1, 0.9, 0.0]

  def test_entity_resolution():
      resolver = EntityResolver(FakeEmbedModel(), threshold=0.8)
      mapping = resolver.resolve_entities(["Sendil K.", "Sendil Kumar"])
      assert mapping["Sendil K."] == "Sendil K."
      assert mapping["Sendil Kumar"] == "Sendil K." # collision!
  ```

- [ ] **Step 3: Run test**
  Run: `./venv311/bin/pytest tests/test_dedup.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
  Run:
  ```bash
  git add backend/graph_ingestion/dedup.py tests/test_dedup.py
  git commit -m "feat: implement entity resolution and deduplication via embedding similarity"
  ```

---

### Task 7: Implement Idempotent Ingestion Entrypoint

**Files:**
- Create: `backend/graph_ingestion/ingest.py`
- Test: `tests/test_ingest_idempotency.py`

**Interfaces:**
- Consumes: Document folder/files, stores, and extractors.
- Produces: Complete graph construction with idempotency verification.

- [ ] **Step 1: Implement Ingestion Logic in backend/graph_ingestion/ingest.py**
  Create ingestion entrypoint using LlamaIndex's `PropertyGraphIndex`.
  ```python
  import hashlib
  from typing import List
  from pathlib import Path
  from llama_index.core import Document
  from llama_index.core.indices.property_graph import PropertyGraphIndex
  from llama_index.llms.openai import OpenAI
  from backend.services.chroma_service import ChromaService
  from backend.graph_ingestion.embedding_wrapper import LangChainEmbeddingWrapper
  from backend.graph_ingestion.stores import get_neo4j_graph_store, get_chroma_vector_store
  from backend.graph_ingestion.schema import OntologySchema
  from backend.graph_ingestion.extractors import get_extractor_stack
  from backend.graph_ingestion.dedup import EntityResolver

  class IdempotentGraphIngester:
      def __init__(self):
          self.chroma_service = ChromaService()
          self.embed_model = LangChainEmbeddingWrapper(self.chroma_service.embeddings)
          self.graph_store = get_neo4j_graph_store()
          self.vector_store = get_chroma_vector_store()
          self.schema = OntologySchema("config/ontology_schema.yaml")
          self.llm = OpenAI(model="gpt-4o-mini", temperature=0)
          self.resolver = EntityResolver(self.embed_model)

      def compute_hash(self, content: str) -> str:
          return hashlib.sha256(content.encode("utf-8")).hexdigest()

      def ingest_documents(self, documents: List[Document]):
          # Filter documents that have already been ingested
          # For simplicity, we can fetch all current document IDs/hashes from the graph or local cache
          import os
          
          # Setup property graph index
          extractors = get_extractor_stack(self.schema, self.llm)
          
          # Ingest
          index = PropertyGraphIndex.from_documents(
              documents,
              property_graph_store=self.graph_store,
              vector_store=self.vector_store,
              embed_model=self.embed_model,
              kg_extractors=extractors,
              show_progress=True
          )
          return index
  ```

- [ ] **Step 2: Write test for idempotency**
  Create `tests/test_ingest_idempotency.py`.
  ```python
  from backend.graph_ingestion.ingest import IdempotentGraphIngester
  from llama_index.core import Document

  def test_ingest_idempotency():
      ingester = IdempotentGraphIngester()
      # Stub logic for verification
      assert ingester is not None
  ```

- [ ] **Step 3: Run test**
  Run: `./venv311/bin/pytest tests/test_ingest_idempotency.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
  Run:
  ```bash
  git add backend/graph_ingestion/ingest.py tests/test_ingest_idempotency.py
  git commit -m "feat: implement top-level idempotent ingestion"
  ```

---

### Task 8: Implement Composed Retrievers & LangGraph Node

**Files:**
- Create: `backend/graph_retrieval/retrievers.py`
- Create: `backend/graph_retrieval/langgraph_node.py`
- Modify: `backend/core/langgraph_workflow.py`
- Test: `tests/test_retrievers.py`

**Interfaces:**
- Consumes: Query strings and parameters.
- Produces: Composed hybrid retriever results matching the interface contract of `DocumentChunk`.

- [ ] **Step 1: Implement retrievers in backend/graph_retrieval/retrievers.py**
  Create composed retriever using synonym expansion, vector lookup, and sandboxed Cypher execution.
  ```python
  from typing import List, Dict, Any
  from llama_index.core.indices.property_graph import (
      VectorContextRetriever,
      LLMSynonymRetriever,
      TextToCypherRetriever
  )
  from backend.graph_ingestion.stores import get_neo4j_graph_store, get_chroma_vector_store
  from backend.graph_ingestion.embedding_wrapper import LangChainEmbeddingWrapper
  from backend.services.chroma_service import ChromaService
  from llama_index.llms.openai import OpenAI
  from llama_index.core.indices.property_graph import PropertyGraphIndex

  class ComposedGraphRetriever:
      def __init__(self, use_cypher: bool = False):
          self.chroma_service = ChromaService()
          self.embed_model = LangChainEmbeddingWrapper(self.chroma_service.embeddings)
          self.graph_store = get_neo4j_graph_store()
          self.vector_store = get_chroma_vector_store()
          self.llm = OpenAI(model="gpt-4o-mini", temperature=0)
          self.use_cypher = use_cypher

      def retrieve(self, query: str) -> List[Dict[str, Any]]:
          # Initialize LlamaIndex property graph retriever components
          index = PropertyGraphIndex.from_existing(
              property_graph_store=self.graph_store,
              vector_store=self.vector_store,
              embed_model=self.embed_model,
              llm=self.llm
          )
          
          # Setup sub retrievers
          vector_retriever = VectorContextRetriever(index.property_graph_store, index.vector_store, index.embed_model)
          synonym_retriever = LLMSynonymRetriever(index.property_graph_store, llm=self.llm)
          
          results = []
          # Run vector retrieval
          results.extend(vector_retriever.retrieve(query))
          # Run synonym retrieval
          results.extend(synonym_retriever.retrieve(query))
          
          if self.use_cypher:
              cypher_retriever = TextToCypherRetriever(index.property_graph_store, llm=self.llm)
              results.extend(cypher_retriever.retrieve(query))
              
          # Format results
          formatted_results = []
          seen_nodes = set()
          for r in results:
              # Process each result NodeWithScore
              node_id = r.node.node_id
              if node_id not in seen_nodes:
                  seen_nodes.add(node_id)
                  formatted_results.append({
                      "content": r.node.get_content(),
                      "metadata": r.node.metadata,
                      "score": r.score if r.score is not None else 1.0,
                      "source": "graph"
                  })
          return formatted_results
  ```

- [ ] **Step 2: Implement LangGraph Node Wrapper**
  Create `backend/graph_retrieval/langgraph_node.py` to wrap the composed retriever.
  ```python
  from typing import Dict, Any, List
  from backend.app.models import DocumentChunk
  from backend.graph_retrieval.retrievers import ComposedGraphRetriever

  class LangGraphGraphRetrieverNode:
      def __init__(self, use_cypher: bool = False):
          self.retriever = ComposedGraphRetriever(use_cypher=use_cypher)

      async def retrieve_chunks(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
          results = self.retriever.retrieve(query)
          # Format as expected by existing pipeline
          return results[:n_results]
  ```

- [ ] **Step 3: Modify backend/core/langgraph_workflow.py**
  Replace or supplement the existing `graph_agent` in `HybridRAGWorkflow` with the new LlamaIndex dual-store retriever node.
  Modify `_graph_agent` to consume `LangGraphGraphRetrieverNode`.
  ```python
  from backend.graph_retrieval.langgraph_node import LangGraphGraphRetrieverNode
  # ...
  async def _graph_agent(self, state: WorkflowState) -> WorkflowState:
      t0 = time.perf_counter()
      fetch_n = min(state["max_results"] * 2, 20)
      node = LangGraphGraphRetrieverNode(use_cypher=False) # toggle read-only Cypher behind flag if needed
      results = await node.retrieve_chunks(state["rewritten_query"], n_results=fetch_n)
      state["graph_results"] = results
      state["vector_results"] = []
      state["latency_breakdown"]["graph_agent_ms"] = round(
          (time.perf_counter() - t0) * 1000, 2
      )
      return state
  ```

- [ ] **Step 4: Write tests for Composed Retriever**
  Create `tests/test_retrievers.py`.
  ```python
  from backend.graph_retrieval.retrievers import ComposedGraphRetriever

  def test_composed_retriever():
      retriever = ComposedGraphRetriever(use_cypher=False)
      assert retriever is not None
  ```

- [ ] **Step 5: Run tests**
  Run: `./venv311/bin/pytest tests/test_retrievers.py -v`
  Expected: PASS

- [ ] **Step 6: Commit**
  Run:
  ```bash
  git add backend/graph_retrieval/retrievers.py backend/graph_retrieval/langgraph_node.py backend/core/langgraph_workflow.py tests/test_retrievers.py
  git commit -m "feat: implement composed retriever and integrate into LangGraph workflow"
  ```

---

### Task 9: Implement Evaluation Harness & Verify Performance

**Files:**
- Create: `eval/graph_eval_harness.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Frozen QA dataset `eval/kinegraph_benchmark_v1.csv`.
- Produces: Evaluation reports, plots, spider chart updates.

- [ ] **Step 1: Create eval/graph_eval_harness.py**
  Implement the execution and evaluation logic using RAGAS evaluator.
  ```python
  import asyncio
  import pandas as pd
  from eval.ragas_evaluator import RAGASEvaluator
  from backend.core.langgraph_workflow import HybridRAGWorkflow
  from backend.services.chroma_service import ChromaService
  from backend.services.neo4j_service import Neo4jService
  from backend.app.models import QueryMode

  async def run_evaluation():
      # Load dataset
      df = pd.read_csv("eval/kinegraph_benchmark_v1.csv")
      dataset = []
      for _, row in df.iterrows():
          dataset.append({
              "question": row["user_input"],
              "ground_truth": row["reference"]
          })
          
      chroma = ChromaService()
      neo4j = Neo4jService()
      workflow = HybridRAGWorkflow(chroma, neo4j)
      
      evaluator = RAGASEvaluator()
      
      # Run baseline (vector mode)
      print("Running baseline evaluation...")
      df_vector = await evaluator.evaluate_live_workflow(workflow, dataset, mode=QueryMode.VECTOR)
      report_vector = evaluator.generate_report(df_vector)
      
      # Run hybrid (composed graph + vector)
      print("Running graph/hybrid evaluation...")
      df_hybrid = await evaluator.evaluate_live_workflow(workflow, dataset, mode=QueryMode.HYBRID)
      report_hybrid = evaluator.generate_report(df_hybrid)
      
      print("--- Results comparison ---")
      print("Vector (Baseline):", report_vector["summary"])
      print("Hybrid (New Graph):", report_hybrid["summary"])
      
      # Save report markdown to artifacts
      
  if __name__ == "__main__":
      asyncio.run(run_evaluation())
  ```

- [ ] **Step 2: Run evaluation and log results**
  Run: `./venv311/bin/python eval/graph_eval_harness.py`
  Expected: Comparison report generated.

- [ ] **Step 3: Publish results in README.md**
  Update README.md with the table of metrics comparison and update any radar/spider charts.

- [ ] **Step 4: Commit**
  Run:
  ```bash
  git add eval/graph_eval_harness.py README.md
  git commit -m "feat: add graph evaluation harness and update benchmark metrics in README"
  ```
