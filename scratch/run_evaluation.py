import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv(os.path.abspath('.env'))

from eval.ragas_evaluator import RAGASEvaluator

# 20-sample evaluation dataset
EVAL_DATASET = [
    {"question": "What is Reciprocal Rank Fusion?",
     "answer": "Reciprocal Rank Fusion (RRF) combines ranked lists from multiple retrieval systems by summing 1/(k+rank) scores.",
     "contexts": ["RRF merges document rankings from heterogeneous retrieval systems using the formula RRF(d) = Σ 1/(k+rank(d)).",
                  "The constant k (default 60) smooths the influence of very high-ranked documents."],
     "ground_truth": "RRF is a rank fusion algorithm that combines multiple ranked lists by summing reciprocal ranks."},

    {"question": "How does ChromaDB store embeddings?",
     "answer": "ChromaDB stores embeddings as dense vectors in its collections along with document text and metadata.",
     "contexts": ["ChromaDB is an open-source vector database that persists embeddings, documents, and metadata in collections.",
                  "Each collection supports HNSW indexing for approximate nearest-neighbour search."],
     "ground_truth": "ChromaDB stores embeddings as dense vectors alongside document text and metadata in named collections."},

    {"question": "What is a knowledge graph?",
     "answer": "A knowledge graph represents entities as nodes and relationships as edges, enabling structured reasoning.",
     "contexts": ["Knowledge graphs encode real-world entities and their interrelations as a directed graph.",
                  "Neo4j is a popular property graph database used to build knowledge graphs."],
     "ground_truth": "A knowledge graph is a structured representation of entities and their relationships."},

    {"question": "What is LangGraph used for?",
     "answer": "LangGraph orchestrates multi-step LLM workflows as stateful directed graphs with conditional edges.",
     "contexts": ["LangGraph is a library for building stateful, multi-actor LLM applications using graph-based control flow.",
                  "It extends LangChain with cycle support and fine-grained state management."],
     "ground_truth": "LangGraph is used to orchestrate stateful multi-step LLM workflows as directed graphs."},

    {"question": "How does RAGAS measure faithfulness?",
     "answer": "RAGAS measures faithfulness by checking what fraction of answer claims are entailed by the retrieved context.",
     "contexts": ["The faithfulness metric decomposes the answer into atomic claims and verifies each against the context.",
                  "A claim is faithful if the context logically implies it."],
     "ground_truth": "Faithfulness measures the fraction of answer claims that are supported by the retrieved context."},

    {"question": "What does context recall measure?",
     "answer": "Context recall measures how much of the ground-truth answer is covered by the retrieved chunks.",
     "contexts": ["Context recall evaluates whether the retrieval system surfaced all information needed to answer the question."],
     "ground_truth": "Context recall measures the fraction of ground-truth information present in the retrieved context."},

    {"question": "What is the role of Celery in the pipeline?",
     "answer": "Celery handles asynchronous document ingestion tasks so the API stays non-blocking.",
     "contexts": ["Celery is a distributed task queue used to offload long-running jobs from the main API process.",
                  "In KineticGraph-Vectra, Celery workers process document uploads asynchronously."],
     "ground_truth": "Celery manages asynchronous background tasks for document ingestion."},

    {"question": "What is hybrid retrieval?",
     "answer": "Hybrid retrieval combines dense (vector) and sparse or structured (graph/BM25) search to improve recall.",
     "contexts": ["Hybrid retrieval fuses results from multiple retrieval paradigms to leverage complementary strengths."],
     "ground_truth": "Hybrid retrieval combines dense vector search with structured or sparse retrieval methods."},

    {"question": "How does Neo4j differ from a relational database?",
     "answer": "Neo4j uses a property-graph model with nodes and edges, making relationship traversal faster than SQL JOINs.",
     "contexts": ["Neo4j stores data as nodes and relationships, enabling efficient graph traversal without JOIN operations."],
     "ground_truth": "Neo4j is a graph database optimised for traversing relationships, unlike relational DBs that use JOINs."},

    {"question": "What is answer relevancy in RAGAS?",
     "answer": "Answer relevancy quantifies how well the generated answer addresses the user's question.",
     "contexts": ["Answer relevancy is computed by generating question variants from the answer and measuring embedding similarity."],
     "ground_truth": "Answer relevancy measures how pertinent the generated answer is to the input question."},

    {"question": "What is vector similarity search?",
     "answer": "Vector similarity search finds the nearest embeddings to a query embedding using distance metrics like cosine similarity.",
     "contexts": ["Approximate nearest-neighbour (ANN) algorithms like HNSW power vector similarity search at scale."],
     "ground_truth": "Vector similarity search retrieves documents whose embeddings are closest to the query embedding."},

    {"question": "Why use RRF over simple score averaging?",
     "answer": "RRF is rank-based and not sensitive to score scale differences between retrieval systems.",
     "contexts": ["Score averaging requires normalisation across systems with different score distributions; RRF avoids this by working on ranks."],
     "ground_truth": "RRF is preferred because it is scale-invariant and handles heterogeneous score distributions."},

    {"question": "What is context precision?",
     "answer": "Context precision measures the signal-to-noise ratio of the retrieved chunks — whether relevant chunks appear at the top.",
     "contexts": ["Context precision evaluates ranking quality by checking if relevant chunks appear before irrelevant ones."],
     "ground_truth": "Context precision measures whether the most relevant retrieved chunks are ranked highest."},

    {"question": "What is a Cypher query?",
     "answer": "Cypher is Neo4j's declarative graph query language for matching node-relationship patterns.",
     "contexts": ["Cypher uses ASCII-art pattern syntax like (a)-[:REL]->(b) to express graph traversal queries."],
     "ground_truth": "Cypher is Neo4j's declarative query language for pattern-matching in property graphs."},

    {"question": "How does LangSmith help in RAG debugging?",
     "answer": "LangSmith captures every step of an LLM chain with inputs, outputs, and latency for root-cause analysis.",
     "contexts": ["LangSmith provides end-to-end tracing of LangChain and LangGraph runs, logging all intermediate states."],
     "ground_truth": "LangSmith traces LLM workflow executions with full input/output and latency data for debugging."},

    {"question": "What does chunk utilisation rate mean?",
     "answer": "Chunk utilisation rate is the fraction of retrieved chunks actually cited in the final generated answer.",
     "contexts": ["A low chunk utilisation rate means the LLM ignores most retrieved context, suggesting retrieval over-fetch."],
     "ground_truth": "Chunk utilisation rate measures how many of the retrieved chunks contribute to the final answer."},

    {"question": "What is the k constant in RRF?",
     "answer": "The k constant (default 60) prevents very-high-ranked documents from dominating the fused score.",
     "contexts": ["In RRF(d) = Σ 1/(k+rank), higher k dampens the impact of top-ranked documents."],
     "ground_truth": "k is a smoothing constant in the RRF formula that controls the influence of top-ranked documents."},

    {"question": "How does FastAPI handle async endpoints?",
     "answer": "FastAPI uses Python's async/await to run endpoints concurrently without blocking the event loop.",
     "contexts": ["FastAPI is built on Starlette and uses asyncio to handle many concurrent requests efficiently."],
     "ground_truth": "FastAPI supports async endpoints via Python's asyncio, enabling high-concurrency without threads."},

    {"question": "What is answer correctness in RAGAS?",
     "answer": "Answer correctness compares the generated answer to a ground-truth reference using semantic similarity and factual overlap.",
     "contexts": ["Answer correctness is a composite metric combining semantic similarity and factual correctness against the ground truth."],
     "ground_truth": "Answer correctness measures semantic and factual similarity between the generated answer and a reference answer."},

    {"question": "What is the purpose of the fusion node in the workflow?",
     "answer": "The fusion node merges vector and graph results using RRF to produce a unified ranked result list.",
     "contexts": ["The fusion_node in the LangGraph workflow applies Reciprocal Rank Fusion to combine vector_agent and graph_agent outputs."],
     "ground_truth": "The fusion node combines results from vector and graph agents using RRF into a single ranked list."},
]

print("Starting evaluation of structured prompt...")
evaluator = RAGASEvaluator(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    metrics=['faithfulness', 'answer_relevancy', 'context_precision',
             'context_recall', 'answer_correctness'],
)

results_df = evaluator.evaluate_batch(EVAL_DATASET[:3])
report = evaluator.generate_report(results_df)

print("\n=== ACTIONABLE RECOMMENDATIONS ===")
for i, rec in enumerate(report['recommendations'], 1):
    print(f'  {i}. {rec}')

print('\n=== QUALITY TIER DISTRIBUTION ===')
for tier, count in report['summary']['quality_distribution'].items():
    pct = count / report['summary']['total_samples'] * 100
    print(f'  {tier:12s}: {count:3d} ({pct:.1f}%)')

print(f"\nOverall Composite Score: {report['summary']['overall_composite_score']:.4f}")
