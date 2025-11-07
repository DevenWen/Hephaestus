"""
Service for generating and comparing embeddings for task deduplication.

This service provides a unified interface for embedding generation using different
providers (OpenAI, OpenRouter) with automatic failover and dimension management.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.core.simple_config import get_config
from src.interfaces.embedding_provider import (
    EmbeddingConfig, EmbeddingProvider, EmbeddingProviderFactory
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and comparing embeddings using configurable providers."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the embedding service.

        Args:
            api_key: API key for embedding provider. If not provided, will be read from config.
        """
        self.config = get_config()
        self._initialize_provider(api_key)
        logger.info(f"Initialized EmbeddingService with provider: {self.provider.config.provider}, model: {self.provider.config.model}")

    def _initialize_provider(self, api_key: Optional[str] = None):
        """Initialize the embedding provider based on configuration."""
        # Use the new configuration attributes
        provider_type = self.config.embedding_provider
        model = self.config.embedding_model
        api_key_env = self.config.embedding_api_key_env
        base_url = self.config.embedding_base_url
        dimensions = self.config.embedding_dimensions
        max_retries = self.config.embedding_max_retries
        timeout = self.config.embedding_timeout

        # Get API key
        if api_key is None:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in environment variable: {api_key_env}")

        # Create provider configuration
        config = EmbeddingConfig(
            provider=provider_type,
            model=model,
            api_key=api_key,
            base_url=base_url,
            dimensions=dimensions,
            max_retries=max_retries,
            timeout=timeout
        )

        # Create provider
        self.provider = EmbeddingProviderFactory.create_provider(config)
        self.dimensions = self.provider.get_dimensions()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using the configured provider.

        Retries up to 3 times with exponential backoff for API errors.

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding vector

        Raises:
            Exception: If embedding generation fails after retries
        """
        try:
            # Truncate text if too long (max ~8000 tokens for most models)
            max_chars = 30000  # Conservative limit
            if len(text) > max_chars:
                logger.warning(f"Text truncated from {len(text)} to {max_chars} characters")
                text = text[:max_chars]

            embedding = await self.provider.generate_embedding(text)
            logger.info(f"Generated embedding with dimension: {len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate multiple embeddings efficiently.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.provider.supports_batch_processing():
            try:
                return await self.provider.generate_embeddings(texts)
            except Exception as e:
                logger.warning(f"Batch embedding generation failed, falling back to individual: {e}")

        # Fallback to individual generation
        embeddings = []
        for text in texts:
            try:
                embedding = await self.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to generate embedding for text: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * self.dimensions)

        return embeddings

    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        # Handle edge cases
        if not vec1 or not vec2:
            logger.warning("Empty vector provided for similarity calculation")
            return 0.0

        if len(vec1) != len(vec2):
            logger.warning(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
            return 0.0

        try:
            # Convert to numpy arrays for efficient computation
            arr1 = np.array(vec1, dtype=np.float32)
            arr2 = np.array(vec2, dtype=np.float32)

            # Calculate norms
            norm_a = np.linalg.norm(arr1)
            norm_b = np.linalg.norm(arr2)

            # Handle zero vectors
            if norm_a == 0 or norm_b == 0:
                logger.warning("Zero vector provided for similarity calculation")
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(arr1, arr2) / (norm_a * norm_b)

            # Ensure result is in valid range (floating point errors can cause slight overflow)
            similarity = np.clip(similarity, -1.0, 1.0)

            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def calculate_batch_similarities(
        self, query_embedding: List[float], embeddings: List[List[float]]
    ) -> List[float]:
        """Calculate cosine similarities between a query and multiple embeddings efficiently.

        Args:
            query_embedding: Query embedding vector
            embeddings: List of embedding vectors to compare against

        Returns:
            List of similarity scores
        """
        if not embeddings:
            return []

        try:
            # Convert to numpy arrays
            query_arr = np.array(query_embedding, dtype=np.float32)
            embeddings_arr = np.array(embeddings, dtype=np.float32)

            # Normalize query
            query_norm = np.linalg.norm(query_arr)
            if query_norm == 0:
                return [0.0] * len(embeddings)
            query_normalized = query_arr / query_norm

            # Normalize embeddings
            norms = np.linalg.norm(embeddings_arr, axis=1)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            embeddings_normalized = embeddings_arr / norms[:, np.newaxis]

            # Calculate dot products (cosine similarities)
            similarities = np.dot(embeddings_normalized, query_normalized)

            # Clip to valid range and convert to list
            similarities = np.clip(similarities, -1.0, 1.0)
            return similarities.tolist()

        except Exception as e:
            logger.error(f"Error in batch similarity calculation: {e}")
            # Fallback to individual calculations
            return [self.calculate_cosine_similarity(query_embedding, emb) for emb in embeddings]

    async def generate_ticket_embedding(
        self, title: str, description: str, tags: List[str]
    ) -> List[float]:
        """
        Generate weighted embedding for ticket content.

        Weighting strategy:
        - Title: 2x weight (repeat title twice in input)
        - Description: 1x weight
        - Tags: 1.5x weight (repeat tags approximately 1.5x)

        Args:
            title: Ticket title
            description: Ticket description
            tags: List of tags

        Returns:
            Embedding vector (dimension depends on configured model)
        """
        # Combine with weights
        # Title gets 2x weight, tags get ~1.5x weight
        tag_text = " ".join(tags)
        weighted_text = f"{title} {title} {description} {tag_text} {tag_text}"

        logger.debug(f"Generating weighted ticket embedding (title 2x, tags 1.5x)")
        return await self.generate_embedding(weighted_text)

    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for search query.

        Args:
            query: Search query text

        Returns:
            Embedding vector (same dimension as ticket embeddings)
        """
        logger.debug(f"Generating query embedding for: {query[:100]}...")
        return await self.generate_embedding(query)

    def get_dimensions(self) -> int:
        """Get the dimension count for embeddings."""
        return self.dimensions

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current embedding provider."""
        return self.provider.get_model_info()

    def normalize_embedding_dimensions(self, embedding: List[float], target_dimensions: int) -> List[float]:
        """Normalize embedding to target dimensions.

        Args:
            embedding: Source embedding vector
            target_dimensions: Target dimension count

        Returns:
            Normalized embedding vector
        """
        current_dims = len(embedding)

        if current_dims == target_dimensions:
            return embedding
        elif current_dims < target_dimensions:
            # Pad with zeros
            return self.provider.pad_embedding(embedding, target_dimensions)
        else:
            # Truncate
            return self.provider.truncate_embedding(embedding, target_dimensions)


# Legacy compatibility function
def create_embedding_service(api_key: str = None) -> EmbeddingService:
    """Create embedding service with legacy signature for backward compatibility."""
    return EmbeddingService(api_key)