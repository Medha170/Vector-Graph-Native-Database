from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- INPUT MODELS (Data Ingestion) ---

class NodeCreate(BaseModel):
    """
    Standard format for adding a node.
    Used by: POST /nodes
    """
    id: str
    text: str
    metadata: Dict[str, Any] = {}

class EdgeCreate(BaseModel):
    """
    Standard format for adding a relationship.
    Used by: POST /edges
    """
    source: str
    target: str
    type: str
    weight: float = 1.0

# --- SEARCH MODELS (Queries) ---

class SearchQuery(BaseModel):
    query_text: str
    top_k: int = 5
    # Weights for the hybrid formula
    vector_weight: float = 0.5
    graph_weight: float = 0.5

class SearchResultItem(BaseModel):
    """
    Uniform response format for all search types.
    """
    id: str
    text: str
    score: float
    reason: str  # e.g., "Vector Similarity: 0.85" or "Graph Neighbor (2 hops)"
    metadata: Dict[str, Any] = {}