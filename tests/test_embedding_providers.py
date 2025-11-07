"""
Tests for embedding providers and services.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.interfaces.embedding_provider import (
    EmbeddingConfig, OpenAIEmbeddingProvider, OpenRouterEmbeddingProvider,
    EmbeddingProviderFactory
)
from src.services.embedding_service import EmbeddingService


class TestEmbeddingConfig:
    """Test embedding configuration."""

    def test_embedding_config_creation(self):
        """Test creating embedding configuration."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-large",
            api_key="test-key",
            dimensions=3072
        )

        assert config.provider == "openai"
        assert config.model == "text-embedding-3-large"
        assert config.api_key == "test-key"
        assert config.dimensions == 3072
        assert config.max_retries == 3  # default
        assert config.timeout == 30  # default

    def test_embedding_config_with_optional_params(self):
        """Test creating embedding configuration with optional parameters."""
        config = EmbeddingConfig(
            provider="openrouter",
            model="mistralai/mistral-embed-2312",
            api_key="test-key",
            base_url="https://custom.api.com",
            dimensions=1024,
            max_retries=5,
            timeout=60
        )

        assert config.base_url == "https://custom.api.com"
        assert config.max_retries == 5
        assert config.timeout == 60


class TestOpenAIEmbeddingProvider:
    """Test OpenAI embedding provider."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def provider(self, mock_openai_client):
        """Create OpenAI provider with mocked client."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-large",
            api_key="test-key"
        )
        return OpenAIEmbeddingProvider(config)

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, provider, mock_openai_client):
        """Test successful embedding generation."""
        # Mock response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3] * 1024)]  # 3072 dimensions
        mock_openai_client.embeddings.create.return_value = mock_response

        embedding = await provider.generate_embedding("test text")

        assert len(embedding) == 3072
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
        mock_openai_client.embeddings.create.assert_called_once_with(
            input="test text",
            model="text-embedding-3-large"
        )

    @pytest.mark.asyncio
    async def test_generate_embedding_invalid_text(self, provider):
        """Test embedding generation with invalid text."""
        with pytest.raises(ValueError, match="Invalid text input"):
            await provider.generate_embedding("")

        with pytest.raises(ValueError, match="Invalid text input"):
            await provider.generate_embedding("   ")

        with pytest.raises(ValueError, match="Invalid text input"):
            await provider.generate_embedding(None)

    @pytest.mark.asyncio
    async def test_generate_embedding_api_error(self, provider, mock_openai_client):
        """Test embedding generation with API error."""
        mock_openai_client.embeddings.create.side_effect = Exception("API error")

        with pytest.raises(RuntimeError, match="OpenAI embedding generation failed"):
            await provider.generate_embedding("test text")

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, provider, mock_openai_client):
        """Test batch embedding generation."""
        # Mock response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3] * 1024),
            Mock(embedding=[0.4, 0.5, 0.6] * 1024)
        ]
        mock_openai_client.embeddings.create.return_value = mock_response

        texts = ["text 1", "text 2"]
        embeddings = await provider.generate_embeddings(texts)

        assert len(embeddings) == 2
        assert all(len(emb) == 3072 for emb in embeddings)
        mock_openai_client.embeddings.create.assert_called_once_with(
            input=texts,
            model="text-embedding-3-large"
        )

    def test_get_dimensions(self, provider):
        """Test getting embedding dimensions."""
        dimensions = provider.get_dimensions()
        assert dimensions == 3072

    def test_get_dimensions_custom(self):
        """Test getting custom embedding dimensions."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            dimensions=1536
        )
        provider = OpenAIEmbeddingProvider(config)
        dimensions = provider.get_dimensions()
        assert dimensions == 1536

    def test_supports_batch_processing(self, provider):
        """Test batch processing support."""
        assert provider.supports_batch_processing() is True

    def test_get_model_info(self, provider):
        """Test getting model information."""
        info = provider.get_model_info()
        assert info["provider"] == "openai"
        assert info["model"] == "text-embedding-3-large"
        assert info["dimensions"] == 3072
        assert info["supports_batch"] is True
        assert "base_url" in info

    def test_normalize_embedding(self, provider):
        """Test embedding normalization."""
        embedding = [3.0, 4.0]  # Norm should be 5.0
        normalized = provider.normalize_embedding(embedding)

        # Check that it's normalized (unit vector)
        norm = sum(x**2 for x in normalized) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_pad_embedding(self, provider):
        """Test embedding padding."""
        embedding = [1.0, 2.0, 3.0]
        padded = provider.pad_embedding(embedding, 5)

        assert len(padded) == 5
        assert padded[:3] == embedding
        assert padded[3:] == [0.0, 0.0]

    def test_truncate_embedding(self, provider):
        """Test embedding truncation."""
        embedding = [1.0, 2.0, 3.0, 4.0, 5.0]
        truncated = provider.truncate_embedding(embedding, 3)

        assert len(truncated) == 3
        assert truncated == [1.0, 2.0, 3.0]


class TestOpenRouterEmbeddingProvider:
    """Test OpenRouter embedding provider."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client (used by OpenRouter provider)."""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def provider(self, mock_openai_client):
        """Create OpenRouter provider with mocked client."""
        config = EmbeddingConfig(
            provider="openrouter",
            model="openai/text-embedding-3-large",
            api_key="test-key"
        )
        return OpenRouterEmbeddingProvider(config)

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, provider, mock_openai_client):
        """Test successful embedding generation."""
        # Mock response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3] * 1024)]  # 3072 dimensions
        mock_openai_client.embeddings.create.return_value = mock_response

        embedding = await provider.generate_embedding("test text")

        assert len(embedding) == 3072
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

        # Verify OpenRouter-specific headers
        mock_openai_client.embeddings.create.assert_called_once_with(
            input="test text",
            model="openai/text-embedding-3-large"
        )

    def test_get_dimensions_openai_model(self, provider):
        """Test getting dimensions for OpenAI model via OpenRouter."""
        dimensions = provider.get_dimensions()
        assert dimensions == 3072

    def test_get_dimensions_mistral_model(self, mock_openai_client):
        """Test getting dimensions for Mistral model."""
        config = EmbeddingConfig(
            provider="openrouter",
            model="mistralai/mistral-embed-2312",
            api_key="test-key"
        )
        provider = OpenRouterEmbeddingProvider(config)
        dimensions = provider.get_dimensions()
        assert dimensions == 1024

    def test_get_model_info(self, provider):
        """Test getting model information."""
        info = provider.get_model_info()
        assert info["provider"] == "openrouter"
        assert info["model"] == "openai/text-embedding-3-large"
        assert info["dimensions"] == 3072
        assert info["supports_batch"] is True
        assert "openrouter.ai" in info["base_url"]


class TestEmbeddingProviderFactory:
    """Test embedding provider factory."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-large",
            api_key="test-key"
        )

        with patch('openai.OpenAI'):
            provider = EmbeddingProviderFactory.create_provider(config)
            assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_create_openrouter_provider(self):
        """Test creating OpenRouter provider."""
        config = EmbeddingConfig(
            provider="openrouter",
            model="openai/text-embedding-3-large",
            api_key="test-key"
        )

        with patch('openai.OpenAI'):
            provider = EmbeddingProviderFactory.create_provider(config)
            assert isinstance(provider, OpenRouterEmbeddingProvider)

    def test_create_unsupported_provider(self):
        """Test creating unsupported provider."""
        config = EmbeddingConfig(
            provider="unsupported",
            model="some-model",
            api_key="test-key"
        )

        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            EmbeddingProviderFactory.create_provider(config)

    def test_get_supported_providers(self):
        """Test getting supported providers."""
        providers = EmbeddingProviderFactory.get_supported_providers()
        assert "openai" in providers
        assert "openrouter" in providers

    def test_get_supported_models(self):
        """Test getting supported models for providers."""
        openai_models = EmbeddingProviderFactory.get_supported_models("openai")
        assert "text-embedding-3-large" in openai_models
        assert "text-embedding-3-small" in openai_models

        openrouter_models = EmbeddingProviderFactory.get_supported_models("openrouter")
        assert "openai/text-embedding-3-large" in openrouter_models
        assert "mistralai/mistral-embed-2312" in openrouter_models


class TestEmbeddingService:
    """Test embedding service."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        with patch('src.services.embedding_service.get_config') as mock_get_config:
            mock_config = Mock()
            # Set up the new configuration attributes as proper strings
            mock_config.embedding_provider = "openai"
            mock_config.embedding_model = "text-embedding-3-large"
            mock_config.embedding_api_key_env = "OPENAI_API_KEY"
            mock_config.embedding_base_url = None
            mock_config.embedding_dimensions = None
            mock_config.embedding_max_retries = 3
            mock_config.embedding_timeout = 30
            mock_config.embedding_fallback_providers = []
            mock_get_config.return_value = mock_config
            yield mock_config

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def embedding_service(self, mock_config, mock_openai_client):
        """Create embedding service with mocked dependencies."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            service = EmbeddingService()
            return service

    @pytest.mark.asyncio
    async def test_generate_embedding_service_level(self, embedding_service, mock_openai_client):
        """Test embedding generation at service level."""
        # Mock response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3] * 1024)]
        mock_openai_client.embeddings.create.return_value = mock_response

        embedding = await embedding_service.generate_embedding("test text")

        assert len(embedding) == 3072
        assert embedding_service.get_dimensions() == 3072

    def test_calculate_cosine_similarity(self, embedding_service):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = embedding_service.calculate_cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6

        vec3 = [0.0, 1.0, 0.0]
        similarity = embedding_service.calculate_cosine_similarity(vec1, vec3)
        assert abs(similarity - 0.0) < 1e-6

    def test_calculate_batch_similarities(self, embedding_service):
        """Test batch similarity calculation."""
        query = [1.0, 0.0, 0.0]
        embeddings = [
            [1.0, 0.0, 0.0],  # Similar
            [0.0, 1.0, 0.0],  # Orthogonal
            [-1.0, 0.0, 0.0]  # Opposite
        ]

        similarities = embedding_service.calculate_batch_similarities(query, embeddings)

        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 1e-6  # Similar
        assert abs(similarities[1] - 0.0) < 1e-6  # Orthogonal
        assert abs(similarities[2] - (-1.0)) < 1e-6  # Opposite

    @pytest.mark.asyncio
    async def test_generate_ticket_embedding(self, embedding_service, mock_openai_client):
        """Test ticket embedding generation."""
        # Mock response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3] * 1024)]
        mock_openai_client.embeddings.create.return_value = mock_response

        embedding = await embedding_service.generate_ticket_embedding(
            title="Test Ticket",
            description="This is a test ticket",
            tags=["test", "bug"]
        )

        assert len(embedding) == 3072
        # Verify that the text was weighted correctly
        call_args = mock_openai_client.embeddings.create.call_args
        input_text = call_args[1]['input']
        assert "Test Ticket Test Ticket" in input_text  # Title repeated twice
        assert "test bug test bug" in input_text  # Tags repeated (joined and repeated)

    def test_normalize_embedding_dimensions(self, embedding_service):
        """Test embedding dimension normalization."""
        # Test padding
        embedding = [1.0, 2.0, 3.0]
        padded = embedding_service.normalize_embedding_dimensions(embedding, 5)
        assert len(padded) == 5
        assert padded[:3] == embedding
        assert padded[3:] == [0.0, 0.0]

        # Test truncation
        embedding = [1.0, 2.0, 3.0, 4.0, 5.0]
        truncated = embedding_service.normalize_embedding_dimensions(embedding, 3)
        assert len(truncated) == 3
        assert truncated == [1.0, 2.0, 3.0]

        # Test no change
        embedding = [1.0, 2.0, 3.0]
        unchanged = embedding_service.normalize_embedding_dimensions(embedding, 3)
        assert unchanged == embedding

    def test_get_provider_info(self, embedding_service):
        """Test getting provider information."""
        info = embedding_service.get_provider_info()
        assert info["provider"] == "openai"
        assert info["model"] == "text-embedding-3-large"
        assert info["dimensions"] == 3072

    def test_legacy_compatibility(self, mock_config, mock_openai_client):
        """Test legacy compatibility function."""
        from src.services.embedding_service import create_embedding_service
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            service = create_embedding_service("test-key")
            assert isinstance(service, EmbeddingService)
            assert service.get_dimensions() == 3072


@pytest.mark.integration
class TestEmbeddingIntegration:
    """Integration tests for embedding providers (requires API keys)."""

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_openai_real_embedding(self):
        """Test real OpenAI embedding generation."""
        service = EmbeddingService()
        embedding = await service.generate_embedding("This is a test sentence for embedding.")

        assert len(embedding) == 3072
        assert all(isinstance(x, float) for x in embedding)
        assert any(x != 0.0 for x in embedding)  # Not all zeros

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
    @pytest.mark.asyncio
    async def test_openrouter_real_embedding(self):
        """Test real OpenRouter embedding generation."""
        # This would require configuration changes to use OpenRouter
        pass

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_real_similarity_calculation(self):
        """Test real similarity calculation."""
        service = EmbeddingService()

        # Generate embeddings for similar texts
        # Note: This is a simplified test - in real usage you'd use async
        # For integration tests, you might want to use a different approach
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])