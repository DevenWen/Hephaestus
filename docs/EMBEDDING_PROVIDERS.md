# Embedding Providers Configuration Guide

This guide explains how to configure and use different embedding providers in Hephaestus, including OpenAI and OpenRouter.

## Overview

Hephaestus now supports multiple embedding providers through a unified interface:
- **OpenAI**: Direct OpenAI API access
- **OpenRouter**: Routing to multiple embedding providers through OpenRouter

## Configuration

### Basic Configuration

Update your `hephaestus_config.yaml` file:

```yaml
# LLM Configuration
llm:
  # Embedding Provider Configuration
  embedding:
    # Primary embedding provider (openai, openrouter)
    provider: "openai"
    # Model to use for embeddings
    model: "text-embedding-3-large"
    # API key environment variable
    api_key_env: "OPENAI_API_KEY"
    # Base URL (optional, for OpenRouter or custom endpoints)
    base_url: null
    # Embedding dimensions (auto-detected from model if not specified)
    dimensions: null
    # Maximum retries for embedding generation
    max_retries: 3
    # Timeout for embedding requests (seconds)
    timeout: 30
    # Alternative providers for failover
    fallback_providers: []
```

### Environment Variables

Set the appropriate API key environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For OpenRouter
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

## Provider-Specific Configurations

### OpenAI Configuration

```yaml
llm:
  embedding:
    provider: "openai"
    model: "text-embedding-3-large"  # 3072 dimensions
    # model: "text-embedding-3-small"  # 1536 dimensions
    # model: "text-embedding-ada-002"  # 1536 dimensions
    api_key_env: "OPENAI_API_KEY"
```

### OpenRouter Configuration

```yaml
llm:
  embedding:
    provider: "openrouter"
    model: "openai/text-embedding-3-large"  # 3072 dimensions (routed through OpenRouter)
    # model: "mistralai/mistral-embed-2312"  # 1024 dimensions
    api_key_env: "OPENROUTER_API_KEY"
    base_url: "https://openrouter.ai/api/v1"
```

## Vector Store Configuration

The vector store now supports dynamic dimensions:

```yaml
vector_store:
  qdrant_url: "http://localhost:6333"
  collection_prefix: "hephaestus"
  # Set to null to auto-detect from embedding provider configuration
  embedding_dimension: null
  # Whether to allow dynamic dimension adjustment during runtime
  allow_dynamic_dimensions: true
```

## Usage Examples

### Basic Usage

```python
from src.services.embedding_service import EmbeddingService

# Initialize service (uses configuration from hephaestus_config.yaml)
embedding_service = EmbeddingService()

# Generate embedding
text = "This is a sample text for embedding generation."
embedding = await embedding_service.generate_embedding(text)

print(f"Generated embedding with {len(embedding)} dimensions")
```

### Advanced Usage with Provider Information

```python
# Get provider information
provider_info = embedding_service.get_provider_info()
print(f"Provider: {provider_info['provider']}")
print(f"Model: {provider_info['model']}")
print(f"Dimensions: {provider_info['dimensions']}")

# Generate multiple embeddings
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = await embedding_service.generate_embeddings(texts)

# Calculate similarities
similarity = embedding_service.calculate_cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity}")
```

### Custom Configuration (Programmatic)

```python
from src.interfaces.embedding_provider import EmbeddingConfig, EmbeddingProviderFactory

# Create custom configuration
config = EmbeddingConfig(
    provider="openrouter",
    model="mistralai/mistral-embed-2312",
    api_key="your-api-key",
    dimensions=1024,
    max_retries=5,
    timeout=60
)

# Create provider
provider = EmbeddingProviderFactory.create_provider(config)

# Use provider directly
embedding = await provider.generate_embedding("Test text")
```

## Migration from OpenAI to OpenRouter

### Automatic Migration

Use the migration utility to migrate existing embeddings:

```bash
# Migrate all collections from OpenAI to OpenRouter
python src/utils/embedding_migration.py \
    --source-provider openai \
    --target-provider openrouter \
    --collections agent_memories static_docs task_completions
```

### Manual Migration

```python
from src.utils.embedding_migration import create_migration_manager

# Create migration manager
migration_manager = await create_migration_manager(
    source_provider="openai",
    target_provider="openrouter",
    target_model="openai/text-embedding-3-large"
)

# Migrate specific collection
stats = await migration_manager.migrate_collection("agent_memories")
print(f"Migration completed: {stats}")
```

## Dimension Management

### Handling Dimension Mismatches

The system automatically handles dimension mismatches:

```python
# Normalize embeddings to target dimensions
source_embedding = [0.1, 0.2, 0.3]  # 3 dimensions
target_dimensions = 5

normalized = embedding_service.normalize_embedding_dimensions(
    source_embedding,
    target_dimensions
)
# Result: [0.1, 0.2, 0.3, 0.0, 0.0] (padded with zeros)
```

### Vector Store Migration

When changing embedding dimensions, you may need to recreate vector collections:

```python
from src.memory.vector_store import VectorStoreManager

# Initialize vector store
vector_store = VectorStoreManager(embedding_dimension=1024)

# Recreate collection with new dimensions
vector_store.recreate_collection("agent_memories", new_dimension=1024)
```

## Provider Comparison

| Provider | Model | Dimensions | Cost per 1M tokens | Context Window |
|----------|--------|------------|-------------------|----------------|
| OpenAI | text-embedding-3-large | 3072 | $0.13 | 8192 |
| OpenAI | text-embedding-3-small | 1536 | $0.02 | 8192 |
| OpenRouter | openai/text-embedding-3-large | 3072 | $0.13 | 8192 |
| OpenRouter | mistralai/mistral-embed-2312 | 1024 | $0.10 | 8192 |

## Best Practices

### 1. Choose the Right Provider
- **OpenAI**: Best for consistent quality and reliability
- **OpenRouter**: Best for cost optimization and provider redundancy

### 2. Dimension Selection
- **3072 dimensions**: Best for high-accuracy semantic search
- **1536 dimensions**: Good balance of accuracy and storage efficiency
- **1024 dimensions**: Most storage-efficient, suitable for large-scale applications

### 3. Migration Strategy
1. Test the new provider with a small subset of data
2. Compare embedding quality and performance
3. Plan for vector store recreation if changing dimensions
4. Implement gradual migration to minimize downtime

### 4. Error Handling
```python
try:
    embedding = await embedding_service.generate_embedding(text)
except RuntimeError as e:
    logger.error(f"Embedding generation failed: {e}")
    # Use fallback provider or zero vector
    embedding = [0.0] * embedding_service.get_dimensions()
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   ValueError: API key not found in environment variable: OPENROUTER_API_KEY
   ```
   Solution: Set the correct environment variable for your chosen provider.

2. **Dimension Mismatches**
   ```
   Warning: Embedding dimension mismatch for 'agent_memories': got 3072, expected 1024
   ```
   Solution: Recreate the vector collection with the correct dimensions.

3. **Provider Not Supported**
   ```
   ValueError: Unsupported embedding provider: custom_provider
   ```
   Solution: Use a supported provider or implement a custom provider class.

### Performance Optimization

1. **Batch Processing**: Use `generate_embeddings()` for multiple texts
2. **Caching**: Cache embeddings for frequently used texts
3. **Dimension Reduction**: Use lower dimensions for faster processing
4. **Provider Failover**: Implement fallback providers for reliability

## Testing

Run the embedding provider tests:

```bash
# Run all embedding tests
pytest tests/test_embedding_providers.py -v

# Run integration tests (requires API keys)
pytest tests/test_embedding_providers.py::TestEmbeddingIntegration -v

# Run specific provider tests
pytest tests/test_embedding_providers.py::TestOpenAIEmbeddingProvider -v
pytest tests/test_embedding_providers.py::TestOpenRouterEmbeddingProvider -v
```

## Configuration Examples

### Development Configuration
```yaml
llm:
  embedding:
    provider: "openai"
    model: "text-embedding-3-small"  # Lower cost for development
    api_key_env: "OPENAI_API_KEY"
```

### Production Configuration
```yaml
llm:
  embedding:
    provider: "openrouter"
    model: "openai/text-embedding-3-large"  # Higher quality for production
    api_key_env: "OPENROUTER_API_KEY"
    max_retries: 5
    timeout: 60
    fallback_providers:
      - provider: "openai"
        model: "text-embedding-3-large"
```

### Cost-Optimized Configuration
```yaml
llm:
  embedding:
    provider: "openrouter"
    model: "mistralai/mistral-embed-2312"  # Lower cost, 1024 dimensions
    api_key_env: "OPENROUTER_API_KEY"
```

This configuration guide should help you effectively use different embedding providers in Hephaestus. For more information, see the API documentation and examples in the `examples/` directory.