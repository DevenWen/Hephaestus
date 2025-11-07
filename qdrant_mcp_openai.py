#!/usr/bin/env python3
"""Custom Qdrant MCP server with multi-provider embeddings.

This is a custom MCP server that wraps Qdrant with configurable embedding providers
(OpenAI and OpenRouter) to match Hephaestus's configuration. By default, it uses the
embedding provider configured in hephaestus_config.yaml (typically OpenRouter).

Supported providers:
- OpenRouter (default): Uses openrouter.ai API with various embedding models
- OpenAI: Uses OpenAI API for embeddings
"""

import os
import sys
import asyncio
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from mcp.server import FastMCP

# Add src to path to import Hephaestus modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.embedding_service import EmbeddingService

# Initialize FastMCP
mcp = FastMCP("Qdrant with Multi-Provider Embeddings")

# Configuration from environment (can be overridden)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hephaestus_agent_memories")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize embedding service (reads from hephaestus_config.yaml)
try:
    embedding_service = EmbeddingService()
    provider_info = embedding_service.get_provider_info()
    print(f"Initialized embedding service with provider: {provider_info['provider']}", file=sys.stderr)
    print(f"  Model: {provider_info['model']}", file=sys.stderr)
    print(f"  Dimensions: {provider_info['dimensions']}", file=sys.stderr)
except Exception as e:
    print(f"Error initializing embedding service: {e}", file=sys.stderr)
    print("Please ensure hephaestus_config.yaml is properly configured and API keys are set", file=sys.stderr)
    sys.exit(1)


async def generate_embedding(text: str) -> List[float]:
    """Generate embedding using configured provider (OpenAI or OpenRouter)."""
    try:
        return await embedding_service.generate_embedding(text)
    except Exception as e:
        raise Exception(f"Failed to generate embedding: {e}")


@mcp.tool()
async def qdrant_find(query: str, limit: int = 5) -> str:
    """Search for relevant information in Qdrant using semantic search.

    Args:
        query: Natural language search query
        limit: Maximum number of results to return (default: 5)

    Returns:
        JSON string with search results containing relevant memories
    """
    try:
        # Generate embedding for query
        query_embedding = await generate_embedding(query)

        # Search Qdrant using query_points (new API)
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
            with_payload=True,
        ).points

        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "score": round(result.score, 4),
                "content": result.payload.get("content", ""),
                "memory_type": result.payload.get("memory_type", "unknown"),
                "agent_id": result.payload.get("agent_id", "unknown"),
                "timestamp": result.payload.get("timestamp", ""),
            })

        if not formatted_results:
            return "No relevant memories found for your query."

        # Format as readable text
        output = f"Found {len(formatted_results)} relevant memories:\n\n"
        for r in formatted_results:
            output += f"[{r['rank']}] Score: {r['score']} | Type: {r['memory_type']}\n"
            output += f"    {r['content']}\n"
            output += f"    (Agent: {r['agent_id'][:8]}... | {r['timestamp'][:10]})\n\n"

        return output

    except Exception as e:
        return f"Error searching Qdrant: {str(e)}"


@mcp.tool()
async def qdrant_store(content: str, metadata: Dict[str, Any] = None) -> str:
    """Store information in Qdrant.

    Note: Agents should use the Hephaestus save_memory tool instead.
    This is provided for completeness but is not the recommended method.

    Args:
        content: Content to store
        metadata: Optional metadata dict

    Returns:
        Success message
    """
    return "Please use the Hephaestus 'save_memory' tool instead of qdrant_store for consistency."


if __name__ == "__main__":
    print(f"Starting Qdrant MCP with Multi-Provider Embeddings", file=sys.stderr)
    print(f"  Provider: {embedding_service.get_provider_info()['provider']}", file=sys.stderr)
    print(f"  Model: {embedding_service.get_provider_info()['model']}", file=sys.stderr)
    print(f"  Dimensions: {embedding_service.get_provider_info()['dimensions']}", file=sys.stderr)
    print(f"  Collection: {COLLECTION_NAME}", file=sys.stderr)
    print(f"  Qdrant: {QDRANT_URL}", file=sys.stderr)

    # Run MCP server
    mcp.run()