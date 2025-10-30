"""Advanced multimodal embedding models with state-of-the-art architectures."""

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

# Advanced embedding models
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagModel, BGEM3FlagModel
from ragatouille import RAGPretrainedModel
import cv2
from PIL import Image
try:
    import whisper
except ImportError:
    # Fallback for systems without whisper
    whisper = None
import librosa

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    async def encode(self, inputs: Union[str, List[str], np.ndarray]) -> np.ndarray:
        """Encode inputs into embeddings."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        pass


class AdvancedTextEmbedding(BaseEmbeddingModel):
    """Advanced text embedding using ColBERT and BGE-M3 models."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize with advanced text embedding models.
        
        Args:
            model_name: Model name (BGE-M3, ColBERT, etc.)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models based on type
        if "colbert" in model_name.lower():
            self.model = RAGPretrainedModel.from_pretrained(
                "colbert-ir/colbertv2.0", verbose=0
            )
        elif "bge-m3" in model_name.lower():
            self.model = BGEM3FlagModel(model_name, use_fp16=True)
        else:
            self.model = SentenceTransformer(model_name, device=self.device)
        
        logger.info(f"Initialized text embedding model: {model_name}")

    async def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts into dense embeddings.
        
        Args:
            texts: Text or list of texts to encode
            
        Returns:
            Dense embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]

        try:
            if hasattr(self.model, 'encode'):
                # BGE-M3 or SentenceTransformer
                if isinstance(self.model, BGEM3FlagModel):
                    embeddings = await asyncio.to_thread(
                        self.model.encode, texts, 
                        batch_size=32, max_length=8192
                    )
                    return embeddings['dense_vecs']
                else:
                    embeddings = await asyncio.to_thread(
                        self.model.encode, texts, 
                        batch_size=32, show_progress_bar=False
                    )
                    return embeddings
            else:
                # ColBERT model
                embeddings = await asyncio.to_thread(
                    self.model.encode, texts
                )
                return np.array(embeddings)

        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    async def encode_sparse(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """Encode texts into sparse embeddings (for BGE-M3).
        
        Args:
            texts: Text or list of texts to encode
            
        Returns:
            Sparse embeddings dictionary
        """
        if isinstance(texts, str):
            texts = [texts]

        if isinstance(self.model, BGEM3FlagModel):
            result = await asyncio.to_thread(
                self.model.encode, texts,
                return_dense=True, return_sparse=True, return_colbert_vecs=True
            )
            return {
                'dense': result['dense_vecs'],
                'sparse': result['lexical_weights'],
                'colbert': result['colbert_vecs']
            }
        else:
            # Fallback to dense only
            dense = await self.encode(texts)
            return {'dense': dense}

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            return self.model.get_sentence_embedding_dimension()
        elif isinstance(self.model, BGEM3FlagModel):
            return 1024  # BGE-M3 dimension
        else:
            return 768  # Default dimension


class AdvancedImageEmbedding(BaseEmbeddingModel):
    """Advanced image embedding using CLIP and other vision models."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        """Initialize image embedding model.
        
        Args:
            model_name: Vision model name
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load vision model
        from transformers import CLIPProcessor, CLIPModel
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        
        logger.info(f"Initialized image embedding model: {model_name}")

    async def encode(self, images: Union[str, Path, Image.Image, List]) -> np.ndarray:
        """Encode images into embeddings.
        
        Args:
            images: Image path, PIL Image, or list of images
            
        Returns:
            Image embeddings array
        """
        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img).convert("RGB")
            processed_images.append(img)

        try:
            inputs = await asyncio.to_thread(
                self.processor, 
                images=processed_images, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                image_features = await asyncio.to_thread(
                    self.model.get_image_features, 
                    **{k: v.to(self.device) for k, v in inputs.items()}
                )
                
            # Normalize embeddings
            embeddings = F.normalize(image_features, p=2, dim=1)
            return embeddings.cpu().numpy()

        except Exception as e:
            logger.error(f"Error encoding images: {e}")
            raise

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.config.projection_dim


class AdvancedAudioEmbedding(BaseEmbeddingModel):
    """Advanced audio embedding using Whisper and audio transformers."""

    def __init__(self, model_name: str = "openai/whisper-large-v3"):
        """Initialize audio embedding model.
        
        Args:
            model_name: Audio model name
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Whisper model
        model_size = model_name.split("-")[-1].replace("v3", "").replace("v2", "")
        if model_size not in ["tiny", "base", "small", "medium", "large"]:
            model_size = "base"
        
        self.model = whisper.load_model(model_size, device=self.device)
        
        logger.info(f"Initialized audio embedding model: {model_name}")

    async def encode(self, audio_paths: Union[str, Path, List]) -> np.ndarray:
        """Encode audio files into embeddings.
        
        Args:
            audio_paths: Audio file path or list of paths
            
        Returns:
            Audio embeddings array
        """
        if not isinstance(audio_paths, list):
            audio_paths = [audio_paths]

        embeddings = []
        
        for audio_path in audio_paths:
            try:
                # Load and preprocess audio
                audio, sr = await asyncio.to_thread(
                    librosa.load, str(audio_path), sr=16000
                )
                
                # Get Whisper embeddings
                result = await asyncio.to_thread(
                    self.model.transcribe, audio, verbose=False
                )
                
                # Extract encoder features as embeddings
                mel = whisper.log_mel_spectrogram(audio).to(self.device)
                with torch.no_grad():
                    encoder_output = self.model.encoder(mel.unsqueeze(0))
                    
                # Pool encoder output to get fixed-size embedding
                pooled_embedding = torch.mean(encoder_output, dim=1).squeeze()
                embeddings.append(pooled_embedding.cpu().numpy())

            except Exception as e:
                logger.error(f"Error encoding audio {audio_path}: {e}")
                # Fallback to zero embedding
                embeddings.append(np.zeros(self.get_embedding_dim()))

        return np.array(embeddings)

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.dims.n_audio_state


class HybridSimilaritySearcher:
    """Advanced similarity search with multiple algorithms."""

    def __init__(self):
        """Initialize similarity searcher."""
        self.logger = logger

    def cosine_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        query_norm = np.linalg.norm(query)
        vectors_norm = np.linalg.norm(vectors, axis=1)
        
        dot_product = np.dot(vectors, query)
        similarity = dot_product / (query_norm * vectors_norm + 1e-8)
        return similarity

    def euclidean_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance-based similarity."""
        distances = np.linalg.norm(vectors - query, axis=1)
        # Convert distance to similarity (0 to 1)
        max_dist = np.max(distances)
        similarity = 1 - (distances / (max_dist + 1e-8))
        return similarity

    def dot_product_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute dot product similarity."""
        return np.dot(vectors, query)

    def manhattan_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute Manhattan distance-based similarity."""
        distances = np.sum(np.abs(vectors - query), axis=1)
        max_dist = np.max(distances)
        similarity = 1 - (distances / (max_dist + 1e-8))
        return similarity

    def angular_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute angular similarity (normalized cosine)."""
        cosine_sim = self.cosine_similarity(query, vectors)
        # Convert to angular distance then to similarity
        angular_dist = np.arccos(np.clip(cosine_sim, -1, 1)) / np.pi
        return 1 - angular_dist

    def hybrid_similarity(
        self, 
        query: np.ndarray, 
        vectors: np.ndarray,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """Compute weighted combination of multiple similarity metrics.
        
        Args:
            query: Query vector
            vectors: Database vectors
            weights: Weights for different similarity metrics
            
        Returns:
            Combined similarity scores
        """
        if weights is None:
            weights = {
                'cosine': 0.4,
                'euclidean': 0.2,
                'dot_product': 0.2,
                'angular': 0.2
            }

        similarities = {}
        
        if weights.get('cosine', 0) > 0:
            similarities['cosine'] = self.cosine_similarity(query, vectors)
            
        if weights.get('euclidean', 0) > 0:
            similarities['euclidean'] = self.euclidean_similarity(query, vectors)
            
        if weights.get('dot_product', 0) > 0:
            similarities['dot_product'] = self.dot_product_similarity(query, vectors)
            
        if weights.get('angular', 0) > 0:
            similarities['angular'] = self.angular_similarity(query, vectors)

        # Normalize each similarity metric to [0, 1]
        for key, sim in similarities.items():
            sim_min, sim_max = np.min(sim), np.max(sim)
            if sim_max > sim_min:
                similarities[key] = (sim - sim_min) / (sim_max - sim_min)

        # Compute weighted combination
        combined_sim = np.zeros(len(vectors))
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in similarities and weight > 0:
                combined_sim += similarities[metric] * weight
                total_weight += weight

        if total_weight > 0:
            combined_sim /= total_weight

        return combined_sim


class MultiModalEmbeddingManager:
    """Manager for all multimodal embedding models."""

    def __init__(self):
        """Initialize the multimodal embedding manager."""
        self.text_embedder = AdvancedTextEmbedding(settings.text_embedding_model)
        self.image_embedder = AdvancedImageEmbedding(settings.image_embedding_model)
        self.audio_embedder = AdvancedAudioEmbedding(settings.audio_embedding_model)
        self.similarity_searcher = HybridSimilaritySearcher()
        
        logger.info("Initialized multimodal embedding manager")

    async def encode_text(self, texts: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """Encode text with multiple representations."""
        dense_embeddings = await self.text_embedder.encode(texts)
        
        # Get sparse embeddings if available
        try:
            sparse_result = await self.text_embedder.encode_sparse(texts)
            return sparse_result
        except:
            return {'dense': dense_embeddings}

    async def encode_image(self, images: Union[str, List]) -> np.ndarray:
        """Encode images."""
        return await self.image_embedder.encode(images)

    async def encode_audio(self, audio_paths: Union[str, List]) -> np.ndarray:
        """Encode audio files."""
        return await self.audio_embedder.encode(audio_paths)

    async def encode_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image]] = None,
        audio: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Encode multiple modalities simultaneously.
        
        Args:
            text: Text content
            image: Image path or PIL Image
            audio: Audio file path
            
        Returns:
            Dictionary with embeddings for each modality
        """
        embeddings = {}
        
        tasks = []
        if text:
            tasks.append(('text', self.encode_text(text)))
        if image:
            tasks.append(('image', self.encode_image(image)))
        if audio:
            tasks.append(('audio', self.encode_audio(audio)))

        # Execute embeddings in parallel
        results = await asyncio.gather(*[task[1] for task in tasks])
        
        # Map results back to modalities
        for i, (modality, _) in enumerate(tasks):
            embeddings[modality] = results[i]

        return embeddings

    def get_embedding_dimensions(self) -> Dict[str, int]:
        """Get embedding dimensions for each modality."""
        return {
            'text': self.text_embedder.get_embedding_dim(),
            'image': self.image_embedder.get_embedding_dim(),
            'audio': self.audio_embedder.get_embedding_dim()
        }

    def search_similar(
        self,
        query_embedding: np.ndarray,
        database_embeddings: np.ndarray,
        similarity_method: str = "hybrid",
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            database_embeddings: Database of embeddings to search
            similarity_method: Similarity computation method
            top_k: Number of top results to return
            
        Returns:
            Tuple of (similarities, indices)
        """
        if similarity_method == "cosine":
            similarities = self.similarity_searcher.cosine_similarity(
                query_embedding, database_embeddings
            )
        elif similarity_method == "euclidean":
            similarities = self.similarity_searcher.euclidean_similarity(
                query_embedding, database_embeddings
            )
        elif similarity_method == "dot_product":
            similarities = self.similarity_searcher.dot_product_similarity(
                query_embedding, database_embeddings
            )
        elif similarity_method == "angular":
            similarities = self.similarity_searcher.angular_similarity(
                query_embedding, database_embeddings
            )
        elif similarity_method == "hybrid":
            similarities = self.similarity_searcher.hybrid_similarity(
                query_embedding, database_embeddings
            )
        else:
            raise ValueError(f"Unknown similarity method: {similarity_method}")

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        return top_similarities, top_indices


# Global embedding manager instance
embedding_manager = MultiModalEmbeddingManager()