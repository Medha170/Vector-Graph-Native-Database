import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

# Import your core logic (NativeDB from core.py)
try:
    from .core import NativeDB
except ImportError:
    from core import NativeDB

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Initialize FastAPI App
app = FastAPI(
    title="Vector+Graph Native Database",
    description="A Hybrid Database supporting Vector Similarity + Graph Connectivity",
    version="1.0"
)

# Enable CORS (Crucial for Frontend/Streamlit communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Instantiate the Database Engine ONCE
# This loads the heavy ML models into memory at startup
db = NativeDB()

# --- Pydantic Models (Validation) ---

class IngestRequest(BaseModel):
    text: str = Field(..., description="Raw unstructured text to process")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source info (e.g., filename, author)")

class SearchRequest(BaseModel):
    query: str = Field(..., description="The search question")
    top_k: int = Field(5, description="Number of results to return")
    mode: str = Field("hybrid", description="Search mode: 'vector', 'graph', or 'hybrid'")

# --- API Endpoints ---

@app.get("/")
def health_check():
    """
    Heartbeat endpoint to check if server is up.
    """
    return {"status": "running", "system": "NativeDB v1.0"}

@app.post("/ingest")
def ingest_data(payload: IngestRequest):
    """
    Ingest Endpoint:
    1. Receives raw text.
    2. Parses it into Nodes and Edges (IngestionPipeline).
    3. Stores it in VectorDB (LanceDB) and GraphDB (SQLite/NetworkX).
    """
    try:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        result = db.ingest(payload.text, source_metadata=payload.metadata)
        return result
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_data(payload: SearchRequest):
    """
    Search Endpoint:
    1. Receives a query.
    2. Performs Hybrid Search (Vector Similarity + Graph Traversal).
    3. Returns ranked results.
    """
    try:
        results = db.search(payload.query, mode=payload.mode, top_k=payload.top_k)
        return {
            "count": len(results),
            "results": results,
            "mode": payload.mode
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph")
def get_graph():
    """
    Visualization Endpoint:
    Returns the full graph structure (nodes/links) for the frontend to render.
    """
    try:
        return db.get_graph_viz()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Server Runner ---
if __name__ == "__main__":
    # Runs the server on localhost:8000 with auto-reload enabled
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)