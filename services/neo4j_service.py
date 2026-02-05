"""
Neo4j Service for Graph Storage
"""
from neo4j import GraphDatabase, Session
from typing import List, Dict, Any, Optional
from core.config import settings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class Neo4jService:
    """Service for interacting with Neo4j Graph Database"""
    
    def __init__(self):
        """Initialize Neo4j driver"""
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        self.llm = ChatOpenAI(
            model="gpt-4",
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0
        )
    
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
        relationships: List[Dict[str, Any]]
    ) -> bool:
        """
        Add a document and its entities/relationships to the graph
        
        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Document metadata
            entities: List of extracted entities
            relationships: List of relationships between entities
            
        Returns:
            Success status
        """
        try:
            with self.driver.session() as session:
                # Create document node
                session.run("""
                    MERGE (d:Document {id: $doc_id})
                    SET d.content = $content,
                        d += $metadata,
                        d.created_at = datetime()
                """, doc_id=doc_id, content=content, metadata=metadata)
                
                # Create entity nodes and link to document
                for entity in entities:
                    session.run("""
                        MERGE (e:Entity {name: $name, type: $type})
                        WITH e
                        MATCH (d:Document {id: $doc_id})
                        MERGE (d)-[:MENTIONS]->(e)
                    """, name=entity['name'], type=entity['type'], doc_id=doc_id)
                
                # Create relationships between entities
                for rel in relationships:
                    session.run("""
                        MATCH (e1:Entity {name: $source})
                        MATCH (e2:Entity {name: $target})
                        MERGE (e1)-[r:RELATES_TO {type: $rel_type}]->(e2)
                        ON CREATE SET r.created_at = datetime()
                    """, source=rel['source'], target=rel['target'], rel_type=rel['type'])
                
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

Cypher Query:
"""
        )
        
        cypher_query = await self.llm.ainvoke(prompt.format(query=natural_language_query))
        return cypher_query.content.strip()
    
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
            
            # Execute query
            with self.driver.session() as session:
                result = session.run(cypher_query)
                
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
