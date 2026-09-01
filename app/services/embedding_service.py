"""Embedding service with thread-safe singleton model loading and async execution."""

import asyncio
import logging
import threading
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Thread-safe Singleton service for generating dense sentence embeddings."""

    _instance: Optional["EmbeddingService"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> "EmbeddingService":
        """Get or initialize the thread-safe singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name=model_name)
        return cls._instance

    def _get_model(self):
        """Lazy load sentence-transformers model thread-safely."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    logger.info(f"Loading embedding model: {self.model_name}...")
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name)
                    logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
        return self._model

    def _encode_sync(self, text: str) -> List[float]:
        """Synchronous CPU-bound encoding call."""
        model = self._get_model()
        # SentenceTransformer encode returns numpy array
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(np.float32).tolist()

    async def generate_embedding(self, text: str) -> List[float]:
        """Asynchronously compute normalized 384-d dense embedding using asyncio.to_thread."""
        return await asyncio.to_thread(self._encode_sync, text)

    @staticmethod
    def embedding_to_bytes(embedding: List[float]) -> bytes:
        """Convert float list embedding to 32-bit float raw bytes for RediSearch vector index."""
        return np.array(embedding, dtype=np.float32).tobytes()


def get_embedding_service(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingService:
    """FastAPI dependency provider for EmbeddingService."""
    return EmbeddingService.get_instance(model_name=model_name)
