# ⚡ HybridDB: Vector + Graph Native Database

A high-performance, **local-first** database that combines **Semantic Understanding (Vector)** with **Structural Reasoning (Graph)** for next-gen AI Retrieval.

---

## 📖 The Problem

Modern RAG (Retrieval Augmented Generation) systems often fail in two ways:

### ❌ Vector Search misses context

It finds "similar words" but doesn't understand relationships (e.g., who created *X*).

### ❌ Graph Search is rigid

It requires exact queries and fails on vague natural language.

---

## ✅ The Solution

A **Native Hybrid Database** that:

* Ingests unstructured text
* Automatically builds a **Knowledge Graph**
* Retrieves information using a **weighted fusion** of

  * Semantic Similarity (vector)
  * Graph Connectivity (structure)

---

## 🚀 Key Features

### 🧠 Automatic Knowledge Graph Construction

Paste raw text → Our SpaCy-powered NLP extracts entities & relationships automatically.

### 🔍 Hybrid Search Engine

Combines **LanceDB (Vector)** + **NetworkX (Graph)** to surface **“Hidden Gems”**:
semantic-distant but structurally relevant nodes.

### ⚡ Zero-Latency Architecture

Runs entirely **locally** using embedded databases (SQLite + LanceDB).
No external API calls.

### 🕸️ Interactive Visualization

Explore your data topology with a physics-based **force-directed graph** viewer.

---

## 🛠️ Tech Stack

| Component      | Technology                   | Why?                                                |
| -------------- | ---------------------------- | --------------------------------------------------- |
| Vector Storage | LanceDB                      | Serverless, disk-based, lightning-fast embeddings   |
| Graph Logic    | NetworkX                     | Powerful in-memory traversal algorithms             |
| Persistence    | SQLite                       | ACID storage for nodes/edges                        |
| Backend API    | FastAPI                      | High-performance async API with auto-generated docs |
| Frontend       | Streamlit                    | Interactive UI + graph visualization                |
| NLP Pipeline   | SpaCy + SentenceTransformers | Robust NER + high-quality embeddings (MiniLM)       |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-team/hybrid-db.git
cd hybrid-db
```

### 2. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Run the System

You need **two terminals** to run the full stack.

#### **Terminal 1: Backend API**

```bash
uvicorn app.main:app --reload
```

Server running at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

#### **Terminal 2: Frontend UI**

```bash
streamlit run frontend/app.py
```

UI running at: **[http://localhost:8501](http://localhost:8501)**

---

## 🧠 The Hybrid Algorithm (How It Works)

HybridDB uses a **Graph-Boosted Semantic Search** algorithm:

### 1. **Anchor Search (Vector)**

Query LanceDB for top-K semantically similar nodes.

Example result:

```
"Elon Musk" (Score: 0.95)
```

### 2. **Context Expansion (Graph)**

Traverse the Knowledge Graph (1–2 hops) from anchor nodes.

Example:

```
Elon Musk → founded → SpaceX
```

### 3. **Score Fusion**

```math
FinalScore = (S_vector × α) + (S_graph_boost × β)
```

Where **α** and **β** are dynamically adjustable in the UI.

This allows surfacing **“SpaceX”** even when not mentioned in the query but connected structurally.

---

## 🔌 API Reference

| Method   | Endpoint  | Description                                                           |
| -------- | --------- | --------------------------------------------------------------------- |
| **POST** | `/ingest` | Accepts raw text, runs ETL, builds graph + vector store               |
| **POST** | `/search` | Performs Hybrid Search (body: `{ "query": "...", "mode": "hybrid" }`) |
| **GET**  | `/graph`  | Returns full node-link graph for visualization                        |

---

### 1. Interactive Knowledge Graph

Visualizes auto-extracted relationships from unstructured text.

### 2. Hybrid Search Results

Displays ranked results based on vector similarity + graph centrality.

---

## 🏆 Hackathon Checklist

* [x] Vector Storage (LanceDB)
* [x] Graph Storage (SQLite/NetworkX)
* [x] Hybrid Retrieval Algorithm
* [x] Local Persistence
* [x] Force-Directed Graph Visualization

---

**Made with 💻 & ☕ for DevForge.