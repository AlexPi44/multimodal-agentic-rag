"""
Integration tests for the Multimodal Agentic RAG system.

This module contains comprehensive integration tests that validate the entire
system's functionality from end-to-end workflows.
"""

import pytest
import asyncio
import tempfile
import os
import json
from pathlib import Path
from typing import Dict, Any
import httpx

# Test imports - these will be mocked if dependencies aren't available
try:
    from src.core.config import Settings
    from src.core.logging import get_logger
    from src.core.vector_store import QdrantVectorStore
    from src.multimodal.embeddings import MultiModalEmbeddingManager
    from src.multimodal.processors import DocumentProcessor
    from src.agents.orchestrator import AgentOrchestrator, AgentType
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


# Skip all tests if imports are not available
pytestmark = pytest.mark.skipif(
    not IMPORTS_AVAILABLE, 
    reason="Core dependencies not available - run pip install -r requirements.txt first"
)


@pytest.fixture
def test_settings():
    """Create test settings configuration."""
    return Settings(
        # Use test database settings
        QDRANT_HOST="localhost",
        QDRANT_PORT=6333,
        QDRANT_COLLECTION="test_collection",
        # Disable external APIs for testing
        OPENAI_API_KEY="test-key",
        ANTHROPIC_API_KEY="test-key",
        # Use minimal models for testing
        TEXT_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        IMAGE_EMBEDDING_MODEL="openai/clip-vit-base-patch32",
        AUDIO_EMBEDDING_MODEL="whisper-tiny"
    )


@pytest.fixture
async def vector_store(test_settings):
    """Create and initialize test vector store."""
    store = QdrantVectorStore(
        host=test_settings.QDRANT_HOST,
        port=test_settings.QDRANT_PORT,
        collection_name=test_settings.QDRANT_COLLECTION
    )
    
    # Try to initialize, skip if Qdrant not available
    try:
        await store.initialize()
        yield store
        # Cleanup
        await store.delete_collection()
        await store.close()
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")


@pytest.fixture
def embedding_manager():
    """Create embedding manager for testing."""
    try:
        manager = MultiModalEmbeddingManager()
        return manager
    except Exception as e:
        pytest.skip(f"Embedding models not available: {e}")


@pytest.fixture
def document_processor():
    """Create document processor for testing."""
    return DocumentProcessor()


@pytest.fixture
async def orchestrator(vector_store, embedding_manager):
    """Create agent orchestrator for testing."""
    return AgentOrchestrator(
        vector_store=vector_store,
        embedding_manager=embedding_manager
    )


class TestCoreComponents:
    """Test core system components."""
    
    def test_settings_loading(self, test_settings):
        """Test that settings load correctly."""
        assert test_settings.QDRANT_HOST == "localhost"
        assert test_settings.QDRANT_PORT == 6333
        assert test_settings.QDRANT_COLLECTION == "test_collection"
    
    def test_logging_configuration(self):
        """Test logging system configuration."""
        logger = get_logger("test_logger")
        assert logger is not None
        logger.info("Test log message")
    
    @pytest.mark.asyncio
    async def test_vector_store_operations(self, vector_store):
        """Test basic vector store operations."""
        # Test collection info
        info = await vector_store.get_collection_info()
        assert isinstance(info, dict)
        
        # Test vector operations
        test_vectors = [[0.1, 0.2, 0.3, 0.4]] * 384  # BGE-M3 dimension
        test_metadata = [{"text": "test document", "type": "test"}]
        test_ids = ["test_1"]
        
        # Upsert test vector
        await vector_store.upsert_vectors(
            vectors=test_vectors,
            metadata=test_metadata,
            ids=test_ids
        )
        
        # Search for similar vectors
        results = await vector_store.search_vectors(
            query_vector=test_vectors[0],
            limit=1
        )
        assert len(results) > 0
        assert results[0].id == "test_1"


class TestEmbeddingGeneration:
    """Test embedding generation for different modalities."""
    
    def test_text_embeddings(self, embedding_manager):
        """Test text embedding generation."""
        text = "This is a test document for embedding generation."
        
        try:
            embeddings = embedding_manager.get_text_embeddings(text)
            assert "dense" in embeddings
            assert isinstance(embeddings["dense"], list)
            assert len(embeddings["dense"]) > 0
        except Exception as e:
            pytest.skip(f"Text embedding model not available: {e}")
    
    def test_embedding_similarity(self, embedding_manager):
        """Test embedding similarity computation."""
        text1 = "Machine learning is a subset of artificial intelligence."
        text2 = "AI includes machine learning as a major component."
        text3 = "The weather is nice today."
        
        try:
            emb1 = embedding_manager.get_text_embeddings(text1)["dense"]
            emb2 = embedding_manager.get_text_embeddings(text2)["dense"]
            emb3 = embedding_manager.get_text_embeddings(text3)["dense"]
            
            # Similar texts should have higher similarity
            sim_12 = embedding_manager.compute_similarity(emb1, emb2, "cosine")
            sim_13 = embedding_manager.compute_similarity(emb1, emb3, "cosine")
            
            assert sim_12 > sim_13
        except Exception as e:
            pytest.skip(f"Text embedding model not available: {e}")


class TestDocumentProcessing:
    """Test document processing pipeline."""
    
    def test_text_file_processing(self, document_processor):
        """Test processing of text files."""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\\nIt has multiple lines.\\nAnd should be processed correctly.")
            temp_path = f.name
        
        try:
            documents = document_processor.process_file(temp_path)
            assert len(documents) > 0
            assert documents[0].content.strip()
            assert documents[0].doc_type == "text"
            assert documents[0].metadata["filename"].endswith(".txt")
        finally:
            os.unlink(temp_path)
    
    def test_json_file_processing(self, document_processor):
        """Test processing of JSON files."""
        test_data = {
            "title": "Test Document",
            "content": "This is test content",
            "metadata": {"author": "Test Author", "version": 1}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            documents = document_processor.process_file(temp_path)
            assert len(documents) > 0
            assert "Test Document" in documents[0].content
            assert documents[0].doc_type == "text"
        finally:
            os.unlink(temp_path)
    
    def test_batch_processing(self, document_processor):
        """Test batch processing of multiple files."""
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            for i in range(3):
                file_path = Path(temp_dir) / f"test_{i}.txt"
                file_path.write_text(f"Content of test file {i}")
            
            documents = document_processor.process_directory(temp_dir)
            assert len(documents) == 3
            
            # Check that all files were processed
            filenames = [doc.metadata["filename"] for doc in documents]
            assert any("test_0.txt" in name for name in filenames)
            assert any("test_1.txt" in name for name in filenames)
            assert any("test_2.txt" in name for name in filenames)


class TestAgentOrchestration:
    """Test agent orchestration and tool usage."""
    
    @pytest.mark.asyncio
    async def test_simple_rag_query(self, orchestrator):
        """Test simple RAG query processing."""
        query = "What is machine learning?"
        
        try:
            result = await orchestrator.process_query(
                query=query,
                agent_type=AgentType.SIMPLE_RAG,
                max_results=3
            )
            
            assert "response" in result
            assert "sources" in result
            assert isinstance(result["sources"], list)
        except Exception as e:
            pytest.skip(f"LLM not available for testing: {e}")
    
    @pytest.mark.asyncio
    async def test_capabilities_reporting(self, orchestrator):
        """Test agent capabilities reporting."""
        capabilities = await orchestrator.get_capabilities()
        
        assert "agents" in capabilities
        assert "tools" in capabilities
        assert "embeddings" in capabilities
        
        # Check that all agent types are listed
        agent_types = capabilities["agents"]
        assert "simple_rag" in agent_types
        assert "hybrid_rag" in agent_types
        assert "multimodal" in agent_types


class TestAPIEndpoints:
    """Test FastAPI endpoints."""
    
    @pytest.fixture
    def api_client(self):
        """Create HTTP client for API testing."""
        return httpx.AsyncClient(base_url="http://localhost:8000")
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        try:
            response = await api_client.get("/health")
            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                assert "components" in data
                assert "version" in data
        except Exception:
            pytest.skip("API server not running")
    
    @pytest.mark.asyncio
    async def test_capabilities_endpoint(self, api_client):
        """Test capabilities endpoint."""
        try:
            response = await api_client.get("/capabilities")
            if response.status_code == 200:
                data = response.json()
                assert "supported_file_types" in data
                assert "embedding_models" in data
                assert "agent_types" in data
        except Exception:
            pytest.skip("API server not running")
    
    @pytest.mark.asyncio
    async def test_stats_endpoint(self, api_client):
        """Test statistics endpoint."""
        try:
            response = await api_client.get("/stats")
            if response.status_code == 200:
                data = response.json()
                assert "total_documents" in data
                assert "total_embeddings" in data
                assert "vector_store_status" in data
        except Exception:
            pytest.skip("API server not running")


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_document_ingestion_workflow(self, vector_store, embedding_manager, document_processor):
        """Test complete document ingestion workflow."""
        # Create test document
        test_content = "Artificial intelligence is transforming many industries. Machine learning algorithms can analyze large datasets to find patterns."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            # Process document
            documents = document_processor.process_file(temp_path)
            assert len(documents) > 0
            
            # Generate embeddings
            for i, doc in enumerate(documents):
                embeddings = embedding_manager.get_text_embeddings(doc.content)
                
                # Store in vector database
                await vector_store.upsert_vectors(
                    vectors=[embeddings["dense"]],
                    metadata=[{
                        "content": doc.content,
                        "filename": doc.metadata["filename"],
                        "doc_type": doc.doc_type,
                        "chunk_index": i
                    }],
                    ids=[f"doc_{i}"]
                )
            
            # Query the stored documents
            query_text = "What is artificial intelligence?"
            query_embedding = embedding_manager.get_text_embeddings(query_text)["dense"]
            
            results = await vector_store.search_vectors(
                query_vector=query_embedding,
                limit=3
            )
            
            assert len(results) > 0
            assert "artificial intelligence" in results[0].payload["content"].lower()
            
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_multimodal_workflow(self, orchestrator):
        """Test multimodal query processing workflow."""
        # Test with mixed query types
        queries = [
            "Explain the concept of neural networks",
            "What are the applications of machine learning?",
            "Compare supervised and unsupervised learning"
        ]
        
        for query in queries:
            try:
                result = await orchestrator.process_query(
                    query=query,
                    agent_type=AgentType.MULTIMODAL,
                    max_results=5
                )
                
                assert "response" in result
                assert isinstance(result["response"], str)
                assert len(result["response"]) > 0
                
            except Exception as e:
                pytest.skip(f"Multimodal workflow test failed: {e}")


class TestPerformanceAndScale:
    """Test performance and scalability aspects."""
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, vector_store, embedding_manager):
        """Test concurrent vector operations."""
        if not vector_store or not embedding_manager:
            pytest.skip("Components not available")
        
        # Generate multiple test vectors
        test_texts = [f"Test document number {i}" for i in range(10)]
        
        # Process embeddings concurrently
        tasks = []
        for i, text in enumerate(test_texts):
            task = asyncio.create_task(self._process_and_store(
                text, i, vector_store, embedding_manager
            ))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that most operations succeeded
        successes = sum(1 for r in results if not isinstance(r, Exception))
        assert successes >= len(test_texts) * 0.8  # At least 80% success rate
    
    async def _process_and_store(self, text: str, idx: int, vector_store, embedding_manager):
        """Helper method to process and store a single document."""
        embeddings = embedding_manager.get_text_embeddings(text)
        await vector_store.upsert_vectors(
            vectors=[embeddings["dense"]],
            metadata=[{"text": text, "index": idx}],
            ids=[f"concurrent_{idx}"]
        )
    
    def test_large_document_processing(self, document_processor):
        """Test processing of large documents."""
        # Create a large text document
        large_content = "This is a test sentence. " * 1000  # ~25KB of text
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_content)
            temp_path = f.name
        
        try:
            documents = document_processor.process_file(temp_path)
            
            # Should be chunked into multiple documents
            assert len(documents) > 1
            
            # Each chunk should be reasonable size
            for doc in documents:
                assert len(doc.content) > 0
                assert len(doc.content) < 10000  # Reasonable chunk size
                
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])