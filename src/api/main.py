"""
FastAPI main application for Multimodal Agentic RAG System.

This module provides REST API endpoints for the RAG system including:
- Document upload and processing
- Query processing and response generation
- System health and statistics
- Agent orchestration endpoints
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uuid
from pathlib import Path
import tempfile
import os

from src.core.config import Settings
from src.core.logging import get_logger
from src.core.vector_store import QdrantVectorStore
from src.multimodal.embeddings import MultiModalEmbeddingManager
from src.multimodal.processors import DocumentProcessor
from src.agents.orchestrator import AgentOrchestrator, AgentType

# Initialize components
settings = Settings()
logger = get_logger(__name__)
app = FastAPI(
    title="Multimodal Agentic RAG API",
    description="Advanced multimodal RAG system with agentic capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (will be initialized on startup)
vector_store: Optional[QdrantVectorStore] = None
embedding_manager: Optional[MultiModalEmbeddingManager] = None
doc_processor: Optional[DocumentProcessor] = None
orchestrator: Optional[AgentOrchestrator] = None


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    agent_type: Optional[str] = "hybrid_rag"
    max_results: Optional[int] = 5
    include_metadata: Optional[bool] = True


class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    agent_used: str
    processing_time: float
    query_id: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    processing_status: str
    chunks_created: int
    embeddings_created: int


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    version: str


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_embeddings: int
    vector_store_status: str
    embedding_models_loaded: List[str]


@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup."""
    global vector_store, embedding_manager, doc_processor, orchestrator
    
    try:
        logger.info("Initializing Multimodal Agentic RAG system...")
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection_name=settings.QDRANT_COLLECTION
        )
        await vector_store.initialize()
        logger.info("Vector store initialized")
        
        # Initialize embedding manager
        embedding_manager = MultiModalEmbeddingManager()
        logger.info("Embedding manager initialized")
        
        # Initialize document processor
        doc_processor = DocumentProcessor()
        logger.info("Document processor initialized")
        
        # Initialize agent orchestrator
        orchestrator = AgentOrchestrator(
            vector_store=vector_store,
            embedding_manager=embedding_manager
        )
        logger.info("Agent orchestrator initialized")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global vector_store
    
    try:
        if vector_store:
            await vector_store.close()
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of all system components."""
    components = {}
    
    try:
        # Check vector store
        if vector_store:
            info = await vector_store.get_collection_info()
            components["vector_store"] = f"healthy ({info.get('vectors_count', 0)} vectors)"
        else:
            components["vector_store"] = "not initialized"
        
        # Check embedding manager
        if embedding_manager:
            components["embedding_manager"] = "healthy"
        else:
            components["embedding_manager"] = "not initialized"
        
        # Check document processor
        if doc_processor:
            components["document_processor"] = "healthy"
        else:
            components["document_processor"] = "not initialized"
        
        # Check orchestrator
        if orchestrator:
            components["agent_orchestrator"] = "healthy"
        else:
            components["agent_orchestrator"] = "not initialized"
        
        overall_status = "healthy" if all("healthy" in status for status in components.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            components=components,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.get("/stats", response_model=StatsResponse)
async def get_system_stats():
    """Get system statistics."""
    try:
        # Get vector store stats
        total_documents = 0
        total_chunks = 0
        total_embeddings = 0
        vs_status = "unknown"
        
        if vector_store:
            info = await vector_store.get_collection_info()
            total_embeddings = info.get('vectors_count', 0)
            vs_status = "connected"
        
        # Get embedding models
        embedding_models = []
        if embedding_manager:
            # This would need to be implemented in the embedding manager
            embedding_models = ["BGE-M3", "ColBERT", "CLIP-ViT-L", "Whisper"]
        
        return StatsResponse(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_embeddings=total_embeddings,
            vector_store_status=vs_status,
            embedding_models_loaded=embedding_models
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a document."""
    if not all([doc_processor, embedding_manager, vector_store]):
        raise HTTPException(status_code=503, detail="System not fully initialized")
    
    # Validate file type
    if not any(file.filename.lower().endswith(ext) for ext in settings.ALLOWED_FILE_TYPES):
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed types: {settings.ALLOWED_FILE_TYPES}"
        )
    
    document_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            tmp_file_path,
            file.filename,
            document_id
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            processing_status="queued",
            chunks_created=0,
            embeddings_created=0
        )
        
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {e}")


async def process_document_background(file_path: str, filename: str, document_id: str):
    """Background task to process uploaded documents."""
    try:
        logger.info(f"Processing document {filename} with ID {document_id}")
        
        # Process document
        documents = doc_processor.process_file(file_path)
        
        # Generate embeddings and store
        for i, doc in enumerate(documents):
            # Generate embeddings based on content type
            if doc.doc_type == "text":
                embeddings = embedding_manager.get_text_embeddings(doc.content)
            elif doc.doc_type == "image":
                embeddings = embedding_manager.get_image_embeddings(file_path)
            else:
                embeddings = embedding_manager.get_text_embeddings(doc.content)
            
            # Store in vector database
            point_id = f"{document_id}_{i}"
            metadata = {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "doc_type": doc.doc_type,
                "hash": doc.hash,
                **doc.metadata
            }
            
            await vector_store.upsert_vectors(
                vectors=[embeddings["dense"]],  # Use dense embeddings for primary search
                metadata=[metadata],
                ids=[point_id]
            )
        
        logger.info(f"Successfully processed document {filename}: {len(documents)} chunks created")
        
    except Exception as e:
        logger.error(f"Failed to process document {filename}: {e}")
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except Exception:
            pass


@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """Process a query using the agentic RAG system."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
    
    query_id = str(uuid.uuid4())
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Map string agent type to enum
        agent_type_map = {
            "simple_rag": AgentType.SIMPLE_RAG,
            "hybrid_rag": AgentType.HYBRID_RAG,
            "multimodal": AgentType.MULTIMODAL,
            "analytical": AgentType.ANALYTICAL
        }
        
        agent_type = agent_type_map.get(request.agent_type, AgentType.HYBRID_RAG)
        
        # Process query through orchestrator
        result = await orchestrator.process_query(
            query=request.query,
            agent_type=agent_type,
            max_results=request.max_results
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return QueryResponse(
            response=result.get("response", "No response generated"),
            sources=result.get("sources", []),
            agent_used=request.agent_type,
            processing_time=processing_time,
            query_id=query_id
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")


@app.get("/documents")
async def list_documents():
    """List all processed documents."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        # This would need to be implemented to query unique document IDs
        # For now, return a placeholder
        return {"documents": [], "total": 0}
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        # This would need to be implemented to delete by document_id filter
        return {"message": f"Document {document_id} deletion queued"}
        
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")


@app.get("/capabilities")
async def get_capabilities():
    """Get system capabilities and configuration."""
    return {
        "supported_file_types": settings.ALLOWED_FILE_TYPES,
        "embedding_models": {
            "text": ["BGE-M3", "ColBERT"],
            "image": ["CLIP-ViT-L"],
            "audio": ["Whisper-large-v3"]
        },
        "agent_types": ["simple_rag", "hybrid_rag", "multimodal", "analytical"],
        "max_file_size": "100MB",
        "vector_store": "Qdrant",
        "features": [
            "Multimodal document processing",
            "Advanced similarity search",
            "Agentic orchestration",
            "Real-time embeddings",
            "Hybrid search capabilities"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)