from typing import List, Dict, Any
from .vector_engine import VectorEngine
from .graph_engine import GraphEngine

class HybridEngine:
    def __init__(self, vector_engine: VectorEngine, graph_engine: GraphEngine):
        self.vec = vector_engine
        self.graph = graph_engine

    def hybrid_search(self, query: str, vector_weight: float = 0.5, graph_weight: float = 0.5, top_k: int = 5):
        """
        The Secret Sauce:
        1. Get initial candidates via Vector Search.
        2. Expand them via Graph Traversal (neighbors).
        3. Re-rank based on formula.
        """
        
        # --- STEP 1: Vector Search (The "Anchor") ---
        # Get more than k (e.g., k*2) to allow graph connections to bubble up
        vector_results = self.vec.search(query, limit=top_k * 2)
        
        # Normalize Vector Scores to 0-1 range (heuristic)
        # LanceDB is already cosine (0-1), but let's ensure it.
        # Map: {node_id: score}
        scores: Dict[str, float] = {}
        for res in vector_results:
            scores[res['id']] = res['score'] * vector_weight

        # --- STEP 2: Graph Expansion (The "Context") ---
        # For every high-ranking vector result, find its neighbors
        # and give them a "Graph Boost"
        
        graph_boosts = {}
        
        for v_res in vector_results:
            node_id = v_res['id']
            # Get neighbors (1 hop)
            neighbors = self.graph.get_neighbors(node_id, depth=1)
            
            for neighbor in neighbors:
                # If neighbor was already found by vector, boost it.
                # If it wasn't, add it as a new candidate.
                
                # Simple decay: Immediate neighbor gets 0.5 * parent_score
                boost = 0.3 * v_res['score'] 
                
                if neighbor in graph_boosts:
                    graph_boosts[neighbor] += boost
                else:
                    graph_boosts[neighbor] = boost

        # --- STEP 3: Merge & Formula ---
        # Final Score = (Vector Score * v_weight) + (Graph Score * g_weight)
        
        final_candidates = {}
        all_ids = set(scores.keys()) | set(graph_boosts.keys())
        
        for nid in all_ids:
            v_score = scores.get(nid, 0.0)
            
            # Normalize graph boost (simple heuristic for hackathon)
            # We cap graph score at 1.0 to prevent explosion
            g_raw = graph_boosts.get(nid, 0.0)
            g_score = min(g_raw, 1.0) * graph_weight
            
            final_score = v_score + g_score
            
            reason = []
            if v_score > 0: reason.append(f"Vector({v_score:.2f})")
            if g_score > 0: reason.append(f"GraphNeighbor({g_score:.2f})")
            
            final_candidates[nid] = {
                "id": nid,
                "score": final_score,
                "reason": " + ".join(reason)
            }

        # --- STEP 4: Sort & Format ---
        sorted_results = sorted(final_candidates.values(), key=lambda x: x['score'], reverse=True)
        
        return sorted_results[:top_k]