from typing import List, Dict, Any
import logging

# Import all your components
try:
    from .ingestion import IngestionPipeline
    from .vector_engine import VectorEngine
    from .graph_engine import GraphEngine
    from .hybrid_logic import HybridEngine
except ImportError:
    from ingestion import IngestionPipeline
    from vector_engine import VectorEngine
    from graph_engine import GraphEngine
    from hybrid_logic import HybridEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NativeDB:
    """
    The One-Stop-Shop for your database.
    Wraps Ingestion, Storage, and Retrieval into a simple API.
    """
    def __init__(self):
        logger.info("Initializing NativeDB...")
        
        # 1. The Parser (Brain)
        self.ingestion = IngestionPipeline()
        
        # 2. The Storage (Memory)
        self.vector_db = VectorEngine()
        self.graph_db = GraphEngine()
        
        # 3. The Logic (Reasoning)
        self.hybrid_engine = HybridEngine(self.vector_db, self.graph_db)
        
        logger.info("NativeDB Ready.")

    def ingest(self, raw_text: str, source_metadata: Dict[str, Any] = None):
        """
        Takes raw text, parses it ONCE, and stores it everywhere.
        """
        if not raw_text.strip():
            return {"status": "ignored", "reason": "empty_text"}

        logger.info("--- Starting Ingestion ---")
        
        # Step 1: Extract Structure (The heavy lifting)
        nodes, edges = self.ingestion.extract_structured_data(raw_text)
        
        # Optional: Enrich nodes with source metadata (e.g., filename, upload date)
        if source_metadata:
            for node in nodes:
                node.metadata.update(source_metadata)

        logger.info(f"Extracted {len(nodes)} Nodes and {len(edges)} Edges.")

        # Step 2: Store in Vector DB (for Semantic Search)
        self.vector_db.add_nodes(nodes)

        # Step 3: Store in Graph DB (for Structural Search)
        self.graph_db.add_nodes(nodes)
        self.graph_db.add_edges(edges)

        return {
            "status": "success",
            "nodes_count": len(nodes),
            "edges_count": len(edges)
        }

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> List[Dict]:
        """
        Unified Search Interface.
        Modes: 'vector', 'graph', 'hybrid'
        """
        if mode == "vector":
            return self.vector_db.search(query, limit=top_k)
        
        elif mode == "graph":
            # Graph-only search usually requires a starting node ID, 
            # but if given text, we first find the node, then traverse.
            # This is essentially hybrid with vector_weight=0.
            return self.hybrid_engine.hybrid_search(query, vector_weight=0.0, graph_weight=1.0, top_k=top_k)
            
        else: # Default to Hybrid
            return self.hybrid_engine.hybrid_search(query, top_k=top_k)

    def get_graph_viz(self):
        """
        Helper for the Frontend to visualize the whole graph.
        """
        return self.graph_db.get_subgraph_json()

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    db = NativeDB()
    
    # 1. Ingest
    text = """
    Python was created by Guido van Rossum. 
    It was first released in 1991.
    """
    db.ingest(text, source_metadata={"source": "history_book"})
    
    # 2. Search
    results = db.search("Who made Python?")
    
    print("\n--- Search Results ---")
    for r in results:
        print(f"[{r['score']:.2f}] {r['id']} ({r.get('reason', 'Match')})")