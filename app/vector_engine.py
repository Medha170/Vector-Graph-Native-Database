import lancedb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
import os
import json  # <--- NEW IMPORT

try:
    from .models import NodeCreate
    from .ingestion import IngestionPipeline
except ImportError:
    from models import NodeCreate
    from ingestion import IngestionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "./data/lancedb_store" 
TABLE_NAME = "nodes"
MODEL_NAME = 'all-MiniLM-L6-v2' 

class VectorEngine:
    def __init__(self):
        os.makedirs(DB_PATH, exist_ok=True)
        logger.info(f"Loading embedding model: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME, device='cpu')
        logger.info(f"Connecting to LanceDB at {DB_PATH}...")
        self.db = lancedb.connect(DB_PATH)
        self.table = None
        self._init_table()

    def _init_table(self):
        if TABLE_NAME in self.db.table_names():
            self.table = self.db.open_table(TABLE_NAME)
            logger.info(f"Vector Table '{TABLE_NAME}' opened.")
        else:
            logger.info(f"Vector Table '{TABLE_NAME}' ready for first insert.")

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def add_nodes(self, nodes: List[NodeCreate]):
        if not nodes: return

        data_to_insert = []
        logger.info(f"Embedding {len(nodes)} nodes...")
        
        for node in nodes:
            vector = self.embed_text(node.text)
            row = {
                "id": node.id,
                "text": node.text,
                "vector": vector,
                "label": node.metadata.get("label", "CONCEPT"),
                # FIX: Serialize dict to JSON string safely
                "metadata": json.dumps(node.metadata) 
            }
            data_to_insert.append(row)

        if self.table is None:
            self.table = self.db.create_table(TABLE_NAME, data=data_to_insert)
        else:
            self.table.add(data_to_insert)
            
        logger.info(f"Stored {len(nodes)} nodes in Vector DB.")

    def search(self, query_text: str, limit: int = 5) -> List[Dict]:
        if self.table is None: return []

        query_vector = self.embed_text(query_text)
        results = self.table.search(query_vector).metric("cosine").limit(limit).to_pandas()

        formatted_results = []
        for _, row in results.iterrows():
            similarity = 1 - row["_distance"]
            
            # FIX: Deserialize JSON string back to dict
            meta_str = row.get("metadata", "{}")
            try:
                meta_dict = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
            except:
                meta_dict = {}

            formatted_results.append({
                "id": row["id"],
                "text": row["text"],
                "score": float(similarity),
                "metadata": meta_dict
            })
            
        return formatted_results