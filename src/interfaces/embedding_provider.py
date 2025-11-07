"""
Embedding Provider Interface for Hephaestus

This module defines the abstract interface for embedding providers and provides
implementations for OpenAI and OpenRouter embedding services.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers"""
    provider: str  # 'openai' or 'openrouter'
    model: str
    api_key: str
    base_url: Optional[str] = None
    dimensions: Optional[int] = None  # Will be inferred from model if not provided
    max_retries: int = 3
    timeout: int = 30


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._dimensions = config.dimensions

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Get the dimension count for this provider's embeddings"""
        pass

    @abstractmethod
    def supports_batch_processing(self) -> bool:
        """Check if this provider supports batch embedding generation"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        pass

    def validate_text(self, text: str) -> bool:
        """Validate text input before embedding generation"""
        if not text or not isinstance(text, str):
            return False
        if len(text.strip()) == 0:
            return False
        return True

    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length"""
        vector = np.array(embedding)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return embedding
        return (vector / norm).tolist()

    def pad_embedding(self, embedding: List[float], target_dimensions: int) -> List[float]:
        """Pad embedding to target dimensions with zeros"""
        current_dims = len(embedding)
        if current_dims >= target_dimensions:
            return embedding[:target_dimensions]

        # Pad with zeros
        padded = embedding + [0.0] * (target_dimensions - current_dims)
        return padded

    def truncate_embedding(self, embedding: List[float], target_dimensions: int) -> List[float]:
        """Truncate embedding to target dimensions"""
        return embedding[:target_dimensions]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation"""

    MODEL_DIMENSIONS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.openai.com/v1",
                max_retries=self.config.max_retries,
                timeout=self.config.timeout,
            )
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        if not self.validate_text(text):
            raise ValueError("Invalid text input")

        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.config.model
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding generation failed: {str(e)}")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate multiple embeddings using OpenAI API"""
        if not texts:
            return []

        # Validate all texts
        for text in texts:
            if not self.validate_text(text):
                raise ValueError(f"Invalid text input: {text}")

        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.config.model
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise RuntimeError(f"OpenAI batch embedding generation failed: {str(e)}")

    def get_dimensions(self) -> int:
        """Get embedding dimensions for the configured model"""
        if self._dimensions:
            return self._dimensions

        # Infer from model name
        if self.config.model in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self.config.model]

        # Default to most common dimension
        return 1536

    def supports_batch_processing(self) -> bool:
        """OpenAI supports batch processing"""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information"""
        return {
            "provider": "openai",
            "model": self.config.model,
            "dimensions": self.get_dimensions(),
            "supports_batch": True,
            "base_url": self.config.base_url or "https://api.openai.com/v1"
        }


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """OpenRouter embedding provider implementation"""

    # OpenRouter routes to different providers, dimensions vary
    SUPPORTED_MODELS = {
        "openai/text-embedding-3-large": 3072,
        "openai/text-embedding-3-small": 1536,
        "mistralai/mistral-embed-2312": 1024,
    }

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenRouter client"""
        try:
            import openai
            # OpenRouter uses OpenAI-compatible API
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://openrouter.ai/api/v1",
                max_retries=self.config.max_retries,
                timeout=self.config.timeout,
                default_headers={
                    "HTTP-Referer": "https://github.com/Hephaestus/Hephaestus",
                    "X-Title": "Hephaestus"
                }
            )
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenRouter API"""
        if not self.validate_text(text):
            raise ValueError("Invalid text input")

        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.config.model
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"OpenRouter embedding generation failed: {str(e)}")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate multiple embeddings using OpenRouter API"""
        if not texts:
            return []

        # Validate all texts
        for text in texts:
            if not self.validate_text(text):
                raise ValueError(f"Invalid text input: {text}")

        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.config.model
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise RuntimeError(f"OpenRouter batch embedding generation failed: {str(e)}")

    def get_dimensions(self) -> int:
        """Get embedding dimensions for the configured model"""
        if self._dimensions:
            return self._dimensions

        # Infer from model name
        if self.config.model in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[self.config.model]

        # Default to most common dimension
        return 1024

    def supports_batch_processing(self) -> bool:
        """OpenRouter supports batch processing"""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenRouter model information"""
        return {
            "provider": "openrouter",
            "model": self.config.model,
            "dimensions": self.get_dimensions(),
            "supports_batch": True,
            "base_url": self.config.base_url or "https://openrouter.ai/api/v1"
        }


class EmbeddingProviderFactory:
    """Factory for creating embedding providers"""

    PROVIDERS = {
        "openai": OpenAIEmbeddingProvider,
        "openrouter": OpenRouterEmbeddingProvider,
    }

    @classmethod
    def create_provider(cls, config: EmbeddingConfig) -> EmbeddingProvider:
        """Create embedding provider based on configuration"""
        if config.provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported embedding provider: {config.provider}")

        provider_class = cls.PROVIDERS[config.provider]
        return provider_class(config)

    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of supported providers"""
        return list(cls.PROVIDERS.keys())

    @classmethod
    def get_supported_models(cls, provider: str) -> List[str]:
        """Get supported models for a provider"""
        if provider == "openai":
            return list(OpenAIEmbeddingProvider.MODEL_DIMENSIONS.keys())
        elif provider == "openrouter":
            return list(OpenRouterEmbeddingProvider.SUPPORTED_MODELS.keys())
        else:
            return []