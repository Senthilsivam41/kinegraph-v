"""
Neo4j Service for Graph Storage
"""
import re

from neo4j import GraphDatabase, READ_ACCESS, Session
from typing import List, Dict, Any, Optional
from backend.core.config import settings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class UnsafeCypherError(ValueError):
    """Raised when generated Cypher exceeds the read-only query contract."""


_CYPHER_LITERAL = re.compile(
    r"'(?:\\.|''|[^'])*'|\"(?:\\.|\"\"|[^\"])*\"|`(?:``|[^`])*`",
    re.DOTALL,
)
_FORBIDDEN_CYPHER = re.compile(
    r"\b(?:CALL|CREATE|DELETE|DETACH|DROP|FOREACH|GRANT|LOAD|MERGE|REMOVE|RENAME|"
    r"REVOKE|SET|SHOW|START|STOP|TERMINATE|USE|ALTER|DENY|SKIP|LIMIT)\b",
    re.IGNORECASE,
)


def validate_read_only_cypher(query: str, result_limit: int) -> str:
    """Validate an LLM query and add the only permitted result bound."""
    if not isinstance(query, str) or not query.strip():
        raise UnsafeCypherError("Generated Cypher is empty")
    if len(query) > 10_000:
        raise UnsafeCypherError("Generated Cypher exceeds the maximum length")
    if not 1 <= result_limit <= 100:
        raise ValueError("result_limit must be between 1 and 100")

    stripped = query.strip()
    masked = _CYPHER_LITERAL.sub(" ", stripped)
    if ";" in masked:
        raise UnsafeCypherError("Multiple Cypher statements are not allowed")
    if "//" in masked or "/*" in masked or "*/" in masked:
        raise UnsafeCypherError("Cypher comments are not allowed")
    if not re.match(r"^(?:OPTIONAL\s+)?MATCH\b", masked, re.IGNORECASE):
        raise UnsafeCypherError("Generated Cypher must start with MATCH or OPTIONAL MATCH")
    if not re.search(r"\bRETURN\b", masked, re.IGNORECASE):
        raise UnsafeCypherError("Generated Cypher must contain RETURN")
    forbidden = _FORBIDDEN_CYPHER.search(masked)
    if forbidden:
        raise UnsafeCypherError(f"Forbidden Cypher clause: {forbidden.group(0).upper()}")
    return f"{stripped}\nLIMIT $result_limit"


class Neo4jService:
    """Service for interacting with Neo4j Graph Database"""
    
    def __init__(self):
        """Initialize Neo4j driver"""
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        openai_key = settings.OPENAI_API_KEY
        kw = {
            "model": "gpt-4",
            "openai_api_key": openai_key,
            "temperature": 0
        }
        if openai_key and (openai_key.startswith("sk-or-") or "openrouter" in openai_key):
            kw["base_url"] = "https://openrouter.ai/api/v1"
        self.llm = ChatOpenAI(**kw)
    
    def close(self):
        """Close the Neo4j driver"""
        if self.driver:
            self.driver.close()
    
    def verify_connectivity(self) -> bool:
        """Verify connection to Neo4j"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                return result.single()[0] == 1
        except Exception as e:
            print(f"Neo4j connectivity error: {e}")
            return False
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        with self.driver.session() as session:
            # Index for documents
            session.run("""
                CREATE INDEX document_id IF NOT EXISTS
                FOR (d:Document) ON (d.id)
            """)
            
            # Index for entities
            session.run("""
                CREATE INDEX entity_name IF NOT EXISTS
                FOR (e:Entity) ON (e.name)
            """)
    
    async def add_document_graph(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        chunks: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Add a document and its entities/relationships to the graph
        
        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Document metadata
            entities: List of extracted entities
            relationships: List of relationships between entities
            chunks: Source chunk bodies already written to ChromaDB
            chunk_ids: ChromaDB identifiers corresponding to ``chunks``
            
        Returns:
            Success status
        """
        try:
            chunks = chunks or []
            chunk_ids = chunk_ids or []
            if len(chunks) != len(chunk_ids):
                raise ValueError("chunks and chunk_ids must have the same length")
            with self.driver.session() as session:
                # Create document node
                session.run("""
                    MERGE (d:Document {id: $doc_id})
                    SET d.content = $content,
                        d += $metadata,
                        d.created_at = datetime()
                """, doc_id=doc_id, content=content, metadata=metadata)

                if chunks:
                    session.run("""
                        UNWIND $chunks AS chunk
                        MERGE (c:__Node__:Chunk {id: chunk.id})
                        SET c.text = chunk.text, c.document_id = $doc_id
                        WITH c
                        MATCH (d:Document {id: $doc_id})
                        MERGE (d)-[:HAS_CHUNK]->(c)
                    """, doc_id=doc_id, chunks=[
                        {"id": chunk_id, "text": chunk}
                        for chunk_id, chunk in zip(chunk_ids, chunks)
                    ])
                
                # Create entity nodes and link to document
                for entity in entities:
                    matching_chunk_ids = [
                        chunk_id for chunk_id, chunk in zip(chunk_ids, chunks)
                        if entity["name"].lower() in chunk.lower()
                    ]
                    if not matching_chunk_ids and chunk_ids:
                        matching_chunk_ids = [chunk_ids[0]]
                    session.run("""
                        MERGE (e:Entity {name: $name, type: $type})
                        WITH e
                        MATCH (d:Document {id: $doc_id})
                        MERGE (d)-[:MENTIONS]->(e)
                        WITH e
                        UNWIND $chunk_ids AS chunk_id
                        MATCH (c:Chunk {id: chunk_id})
                        MERGE (c)-[:MENTIONS]->(e)
                    """, name=entity['name'], type=entity['type'], doc_id=doc_id, chunk_ids=matching_chunk_ids)
                
                # Create relationships between entities
                for rel in relationships:
                    evidence = rel.get("evidence_text") or rel.get("evidence") or (
                        f"{rel['source']} {rel['type']} {rel['target']} was extracted from document {doc_id}."
                    )
                    session.run("""
                        MATCH (e1:Entity {name: $source})
                        MATCH (e2:Entity {name: $target})
                        MERGE (e1)-[r:RELATES_TO {type: $rel_type}]->(e2)
                        ON CREATE SET r.created_at = datetime()
                        SET r.evidence_text = $evidence_text,
                            r.weight = $weight
                    """, source=rel['source'], target=rel['target'], rel_type=rel['type'],
                         evidence_text=evidence, weight=float(rel.get("weight", 0.75)))

            self.last_enrichment_result = None
            if chunk_ids:
                from backend.graph_ingestion.enrichment import NodeEnricher
                from backend.services.chroma_service import ChromaService

                chroma = ChromaService()
                self.last_enrichment_result = NodeEnricher(
                    self.driver, chroma.client
                ).enrich(chunk_ids=chunk_ids)
            return True
        except Exception as e:
            print(f"Error adding document to Neo4j: {e}")
            return False
    
    async def query_to_cypher(self, natural_language_query: str) -> str:
        """
        Convert natural language query to Cypher using LLM
        
        Args:
            natural_language_query: User's query in natural language
            
        Returns:
            Generated Cypher query
        """
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
You are a Neo4j Cypher query expert. Convert the following natural language query to a Cypher query.

Schema:
- Nodes: Document (id, content, metadata), Entity (name, type)
- Relationships: MENTIONS (Document->Entity), RELATES_TO (Entity->Entity)

Natural Language Query: {query}

Generate ONLY the Cypher query without any explanation. The query should return relevant information.
Use a read-only MATCH/OPTIONAL MATCH query with RETURN. Do not use CALL, write clauses,
comments, semicolons, parameters, SKIP, or LIMIT; the application applies its own result limit.

Cypher Query:
"""
        )
        
        cypher_query = await self.llm.ainvoke(prompt.format(query=natural_language_query))
        query_str = cypher_query.content.strip()
        if query_str.startswith("```"):
            lines = query_str.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            query_str = "\n".join(lines).strip()
        return query_str
    
    async def graph_search(
        self,
        query: str,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform graph search using natural language
        
        Args:
            query: Natural language query
            n_results: Maximum number of results
            
        Returns:
            List of results from graph query
        """
        try:
            # Convert to Cypher
            cypher_query = await self.query_to_cypher(query)
            safe_query = validate_read_only_cypher(cypher_query, n_results)
            
            # Execute in read access mode after deterministic clause validation.
            with self.driver.session(default_access_mode=READ_ACCESS) as session:
                result = session.run(safe_query, result_limit=n_results)
                
                formatted_results = []
                for record in result:
                    # Extract document or entity information
                    result_dict = {}
                    for key in record.keys():
                        value = record[key]
                        if hasattr(value, '__dict__'):
                            result_dict[key] = dict(value)
                        else:
                            result_dict[key] = value
                    
                    # Format for consistency with vector results
                    if 'd' in result_dict and 'content' in result_dict['d']:
                        formatted_results.append({
                            'content': result_dict['d']['content'],
                            'metadata': {k: v for k, v in result_dict['d'].items() if k != 'content'},
                            'score': 1.0,  # Graph results don't have distance scores
                            'source': 'graph'
                        })
                    
                    if len(formatted_results) >= n_results:
                        break
                
                return formatted_results
                
        except UnsafeCypherError as exc:
            print(f"Rejected unsafe generated Cypher: {exc}")
            raise
        except Exception as e:
            print(f"Error performing graph search: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get the number of documents in the graph"""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (d:Document) RETURN count(d) as count")
                return result.single()['count']
        except Exception:
            return 0
    
    def clear_database(self):
        """Clear all nodes and relationships (USE WITH CAUTION)"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                return True
        except Exception:
            return False
