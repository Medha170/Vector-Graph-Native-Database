import networkx as nx
import sqlite3
import os
import logging
from typing import List, Dict, Set, Any
import json

# Robust imports for models and ingestion pipeline
try:
    from .models import NodeCreate, EdgeCreate
except ImportError:
    from models import NodeCreate, EdgeCreate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "./data/graph.sqlite"

class GraphEngine:
    def __init__(self):
        """
        Hybrid Storage:
        - SQLite for persistence (saving to disk).
        - NetworkX for logic (traversal, pathfinding).
        """
        os.makedirs("./data", exist_ok=True)
        self.graph = nx.DiGraph()
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._load_graph_from_db()

    def _init_db(self):
        """
        Creates SQLite tables if they don't exist.
        """
        cursor = self.conn.cursor()
        
        # Nodes Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                metadata TEXT
            )
        """)
        
        # Edges Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT,
                target TEXT,
                type TEXT,
                weight REAL,
                PRIMARY KEY (source, target, type)
            )
        """)
        self.conn.commit()

    def _load_graph_from_db(self):
        """
        Hydrates the NetworkX graph from SQLite on startup.
        """
        logger.info("Hydrating Graph from SQLite...")
        cursor = self.conn.cursor()
        
        # Load Nodes
        cursor.execute("SELECT id, metadata FROM nodes")
        nodes = cursor.fetchall()
        for n_id, meta in nodes:
            # Safely handle potential JSON errors
            try:
                meta_dict = json.loads(meta)
            except:
                meta_dict = {}
            self.graph.add_node(n_id, **meta_dict)
            
        # Load Edges
        cursor.execute("SELECT source, target, type, weight FROM edges")
        edges = cursor.fetchall()
        for src, tgt, type_, weight in edges:
            self.graph.add_edge(src, tgt, type=type_, weight=weight)
            
        logger.info(f"Graph hydrated: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def add_nodes(self, nodes: List[NodeCreate]):
        """
        Persist nodes to SQLite and update memory.
        """
        if not nodes:
            return

        cursor = self.conn.cursor()
        for node in nodes:
            # 1. Update In-Memory Graph
            self.graph.add_node(node.id, **node.metadata)
            
            # 2. Persist to SQLite
            cursor.execute(
                "INSERT OR REPLACE INTO nodes (id, metadata) VALUES (?, ?)",
                (node.id, json.dumps(node.metadata))
            )
        self.conn.commit()
        logger.info(f"Added {len(nodes)} nodes to Graph.")

    def add_edges(self, edges: List[EdgeCreate]):
        """
        Persist edges to SQLite and update memory.
        """
        if not edges:
            return

        cursor = self.conn.cursor()
        for edge in edges:
            # 1. Update In-Memory Graph
            self.graph.add_edge(edge.source, edge.target, type=edge.type, weight=edge.weight)
            
            # 2. Persist to SQLite
            cursor.execute(
                "INSERT OR REPLACE INTO edges (source, target, type, weight) VALUES (?, ?, ?, ?)",
                (edge.source, edge.target, edge.type, edge.weight)
            )
        self.conn.commit()
        logger.info(f"Added {len(edges)} edges to Graph.")

    def get_neighbors(self, node_id: str, depth: int = 1) -> List[str]:
        """
        Retrieval: Finds neighbors (INCOMING and OUTGOING).
        """
        if node_id not in self.graph:
            return []
            
        if depth == 1:
            # Look both ways! (Successors = Outgoing, Predecessors = Incoming)
            outgoing = list(self.graph.successors(node_id))
            incoming = list(self.graph.predecessors(node_id))
            return list(set(outgoing + incoming))
        
        # BFS for deeper hops (convert to undirected temporarily for traversal)
        undirected_view = self.graph.to_undirected()
        paths = nx.single_source_shortest_path_length(undirected_view, node_id, cutoff=depth)
        return [n for n in paths.keys() if n != node_id]

    def get_subgraph_json(self):
        """
        Returns graph data compatible with visualization libraries (e.g., Pyvis/Streamlit).
        Ensures the key 'links' is present instead of 'edges'.
        """
        data = nx.node_link_data(self.graph)
        
        # FIX: Rename 'edges' to 'links' to satisfy Test Case & D3 Standard
        if "edges" in data:
            data["links"] = data.pop("edges")
            
        return data