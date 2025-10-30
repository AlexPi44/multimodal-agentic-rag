"""Vector database implementation using Qdrant."""

import uuid
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Vector database interface using Qdrant."""

    def __init__(self, collection_name: str = "multimodal_vectors"):
        """Initialize the vector store."""
        self.collection_name = collection_name
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client with error handling."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
            
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=30
            )
            self._setup_collection()
            logger.info("Vector store initialized successfully")
        except ImportError:
            logger.warning("Qdrant client not available, using mock implementation")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            self.client = None

    def _setup_collection(self):
        """Setup the collection with proper vector configuration."""
        if not self.client:
            return
            
        try:
            from qdrant_client.http.models import Distance, VectorParams
            
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name for col in collections.collections
            )

            if not collection_exists:
                # Create collection with multiple vector fields for multimodal data
                vector_params = {
                    "text": VectorParams(size=1024, distance=Distance.COSINE),
                    "image": VectorParams(size=512, distance=Distance.COSINE),
                    "audio": VectorParams(size=768, distance=Distance.COSINE),
                }

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_params
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")

    async def upsert(
        self,
        vectors: Dict[str, List[float]],
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
    ) -> str:
        """Insert or update vectors in the database."""
        if document_id is None:
            document_id = str(uuid.uuid4())

        if not self.client:
            logger.warning("Vector store not available, returning mock ID")
            return document_id

        try:
            from qdrant_client.http import models
            
            point = models.PointStruct(
                id=document_id,
                vector=vectors,
                payload=metadata
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            logger.info(f"Upserted document {document_id}")
        except Exception as e:
            logger.error(f"Failed to upsert document: {e}")

        return document_id

    async def search(
        self,
        query_vector: List[float],
        vector_type: str = "text",
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self.client:
            logger.warning("Vector store not available, returning empty results")
            return []

        try:
            from qdrant_client.http import models
            
            search_filter = None
            if filter_conditions:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(key=key, match=models.MatchValue(value=value))
                        for key, value in filter_conditions.items()
                    ]
                )

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=(vector_type, query_vector),
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            formatted_results = [
                {
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.payload,
                }
                for result in results
            ]
            
            logger.info(f"Found {len(formatted_results)} similar vectors")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def hybrid_search(
        self,
        text_vector: Optional[List[float]] = None,
        image_vector: Optional[List[float]] = None,
        audio_vector: Optional[List[float]] = None,
        weights: Optional[Dict[str, float]] = None,
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search across multiple vector types."""
        if weights is None:
            weights = {"text": 1.0, "image": 1.0, "audio": 1.0}

        all_results = {}

        # Search each vector type and collect results
        for vector_type, vector in [
            ("text", text_vector),
            ("image", image_vector), 
            ("audio", audio_vector),
        ]:
            if vector is not None:
                results = await self.search(
                    query_vector=vector,
                    vector_type=vector_type,
                    limit=limit * 2,  # Get more results for better fusion
                    filter_conditions=filter_conditions,
                )

                weight = weights.get(vector_type, 1.0)
                for result in results:
                    doc_id = result["id"]
                    if doc_id not in all_results:
                        all_results[doc_id] = {
                            "id": doc_id,
                            "metadata": result["metadata"],
                            "scores": {},
                            "combined_score": 0.0,
                        }
                    all_results[doc_id]["scores"][vector_type] = result["score"] * weight

        # Combine scores and sort
        for result in all_results.values():
            result["combined_score"] = sum(result["scores"].values())

        # Sort by combined score and return top results
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["combined_score"],
            reverse=True,
        )

        logger.info(f"Hybrid search returned {len(sorted_results[:limit])} results")
        return sorted_results[:limit]

    async def delete(self, document_id: str) -> bool:
        """Delete a document by ID."""
        if not self.client:
            logger.warning("Vector store not available")
            return False

        try:
            from qdrant_client.http import models
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[document_id])
            )
            logger.info(f"Deleted document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self.client:
            return {"error": "Vector store not available"}

        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "vector_config": str(info.config.params.vectors),
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    def close(self):
        """Close the database connection."""
        # Qdrant client doesn't need explicit closing
        pass


# Global vector store instance
vector_store = VectorStore()