import re
import json
import logging
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. ROBUST IMPORTS ---
try:
    from .models import NodeCreate, EdgeCreate
except ImportError:
    from models import NodeCreate, EdgeCreate

# --- 2. OPTIONAL DEPENDENCIES ---
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logger.warning("Trafilatura not found. HTML parsing will be basic.")

try:
    from ftfy import fix_text
except ImportError:
    fix_text = lambda x: x  

try:
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
except Exception as e:
    UNSTRUCTURED_AVAILABLE = False

import spacy

# --- 3. HANDLER INTERFACES ---

class Handler:
    name: str
    def applies(self, text: str) -> bool: raise NotImplementedError
    def process(self, text: str) -> str: raise NotImplementedError

class HandlerRegistry:
    def __init__(self):
        self._handlers: List[Handler] = []
    def register(self, handler: Handler):
        self._handlers.append(handler)
    def find(self, text: str) -> Optional[Handler]:
        for h in self._handlers:
            try:
                if h.applies(text): return h
            except Exception: continue
        return None

# --- 4. CONCRETE HANDLERS ---

class HTMLHandler(Handler):
    name = "html"
    def applies(self, text: str) -> bool:
        return "<" in text and ">" in text and ("<html" in text.lower() or "<div" in text.lower())
    def process(self, text: str) -> str:
        if TRAFILATURA_AVAILABLE:
            extracted = trafilatura.extract(text)
            return extracted if extracted else text
        else:
            return re.sub(r'<[^>]+>', '', text)

class JSONHandler(Handler):
    name = "json"
    def applies(self, text: str) -> bool:
        t = text.strip()
        return (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]"))
    def process(self, text: str) -> str:
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, indent=2)
        except: return text

class UnstructuredHandler(Handler):
    name = "unstructured"
    def applies(self, text: str) -> bool:
        return UNSTRUCTURED_AVAILABLE and len(text) > 50
    def process(self, text: str) -> str:
        try:
            elements = partition(text=text)
            return "\n\n".join([str(e) for e in elements])
        except Exception: return text

# --- 5. MAIN PIPELINE ---

class IngestionPipeline:
    def __init__(self):
        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.registry = HandlerRegistry()
        self.registry.register(HTMLHandler())
        self.registry.register(JSONHandler())
        if UNSTRUCTURED_AVAILABLE:
            self.registry.register(UnstructuredHandler())

    def preprocess(self, text: str) -> str:
        # 1. Format Handling
        handler = self.registry.find(text)
        if handler:
            text = handler.process(text)
        
        # 2. Aggressive Cleaning
        text = fix_text(text)
        # Remove citations like [1], [citation needed]
        text = re.sub(r'\[.*?\]', '', text)
        # Remove parentheses content just for cleaner graph nodes? 
        # Actually better to keep them in text but clean them in extraction.
        
        text = " ".join(text.split()) 
        return text

    def extract_structured_data(self, text: str) -> Tuple[List[NodeCreate], List[EdgeCreate]]:
        cleaned_text = self.preprocess(text)
        doc = self.nlp(cleaned_text)

        nodes: Dict[str, NodeCreate] = {}
        edges = []

        # Helper to clean node IDs
        def clean_node_id(text):
            # Remove (stuff), [stuff], and extra spaces
            text = re.sub(r'\(.*?\)', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            return text.strip()

        # --- NER (Nodes) ---
        for ent in doc.ents:
            if ent.label_ in ["CARDINAL", "DATE", "TIME", "PERCENT", "QUANTITY", "ORDINAL"]:
                continue
            
            e_text = clean_node_id(ent.text)
            if len(e_text) < 2: continue
            
            if e_text not in nodes:
                nodes[e_text] = NodeCreate(
                    id=e_text,
                    text=f"{ent.label_}: {ent.text}",
                    metadata={"label": ent.label_}
                )

        # --- RELATION EXTRACTION ---
        def expand_noun_phrase(token):
            items = [token]
            for child in token.children:
                if child.dep_ == "conj":
                    items.extend(expand_noun_phrase(child))
            return items

        for token in doc:
            if token.pos_ in ("VERB", "AUX"):
                subjects = []
                objects = []
                
                # 1. Subjects
                for child in token.children:
                    if child.dep_ in ("nsubj", "csubj", "nsubjpass", "csubjpass"):
                        subjects.extend(expand_noun_phrase(child))
                
                # ACL support
                if not subjects and token.dep_ in ("acl", "relcl", "advcl"):
                    subjects.append(token.head)

                # 2. Objects
                for child in token.children:
                    if child.dep_ in ("dobj", "attr", "acomp", "oprd"):
                        objects.extend(expand_noun_phrase(child))
                    if child.dep_ == "prep":
                        for grand_child in child.children:
                            if grand_child.dep_ == "pobj":
                                objects.extend(expand_noun_phrase(grand_child))
                    if child.dep_ == "agent":
                         for grand_child in child.children:
                            if grand_child.dep_ == "pobj":
                                objects.extend(expand_noun_phrase(grand_child))

                # 3. Create Edges
                for s_token in subjects:
                    for o_token in objects:
                        s_text = clean_node_id(s_token.text)
                        o_text = clean_node_id(o_token.text)
                        
                        # Remove determiners
                        s_text = re.sub(r'^(The|A|An)\s+', '', s_text, flags=re.IGNORECASE)
                        o_text = re.sub(r'^(The|A|An)\s+', '', o_text, flags=re.IGNORECASE)

                        # Strict Filters
                        if len(s_text) < 2 or len(o_text) < 2: continue
                        
                        # BLOCK PRONOUNS / GENERIC WORDS
                        stop_concepts = ["it", "he", "she", "they", "this", "that", "one", "who", "which"]
                        if s_text.lower() in stop_concepts or o_text.lower() in stop_concepts:
                            continue

                        # Register Nodes
                        if s_text not in nodes:
                            nodes[s_text] = NodeCreate(id=s_text, text=s_text, metadata={"label": "CONCEPT"})
                        if o_text not in nodes:
                            nodes[o_text] = NodeCreate(id=o_text, text=o_text, metadata={"label": "CONCEPT"})
                        
                        # Add Edge
                        edges.append(EdgeCreate(
                            source=s_text,
                            target=o_text,
                            type=token.lemma_,
                            weight=1.0
                        ))

        return list(nodes.values()), edges