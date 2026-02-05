"""
ChromaDB Service for Vector Storage
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from core.config import settings


class ChromaService:
    """Service for interacting with ChromaDB"""
    
    def __init__(self):
        """Initialize ChromaDB client"""
        self.client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        self.collection_name = settings.CHROMA_COLLECTION_NAME
    
    def get_or_create_collection(self):
        """Get or create the collection"""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "KineticGraph-Vectra document embeddings"}
        )
    
    async def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """
        Add documents to ChromaDB
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dictionaries
            ids: List of unique IDs
            
        Returns:
            Success status
        """
        try:
            collection = self.get_or_create_collection()
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return True
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            return False
    
    async def similarity_search(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search
        
        Args:
            query: Query text
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of results with documents, metadata, and distances
        """
        try:
            collection = self.get_or_create_collection()
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters if filters else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'score': 1 / (1 + results['distances'][0][i]),  # Convert distance to score
                        'source': 'vector'
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []
    
    def delete_collection(self):
        """Delete the collection (useful for testing)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            return True
        except Exception:
            return False
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        try:
            collection = self.get_or_create_collection()
            return collection.count()
        except Exception:
            return 0
