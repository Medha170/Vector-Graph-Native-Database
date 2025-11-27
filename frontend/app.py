import streamlit as st
import requests
import pandas as pd
import visualizer # Imports the logic from visualizer.py

# --- CONFIG ---
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="HybridDB", layout="wide", page_icon="⚡")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Dark Mode Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1E212B;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    div[data-testid="stMetric"] label {
        color: #A0A0A0 !important; 
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important; 
    }
    
    /* Clean Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #FFFFFF;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0E1117;
        border-bottom: 2px solid #FF4B8B;
        color: #FF4B8B;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("⚡ Vector + Graph Native Database")
st.markdown("A Hybrid RAG System for **Semantic Understanding** and **Structural Reasoning**.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    search_mode = st.radio("Search Mode", ["hybrid", "vector", "graph"], index=0)
    top_k = st.slider("Top K Results", 1, 20, 5)
    st.markdown("---")
    
    # Simple Health Check Indicator
    try:
        requests.get(f"{API_URL}/", timeout=1)
        st.success("Backend Connected")
    except:
        st.error("Backend Offline")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📥 Ingestion", "🕸️ Graph Explorer"])

# ==========================================
# TAB 1: SEARCH
# ==========================================
with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Ask a question:", placeholder="e.g., Who created Python?")
    with col2:
        st.write("") 
        st.write("") 
        search_btn = st.button("Search", type="primary", use_container_width=True)

    if search_btn and query:
        with st.spinner("Running Hybrid Retrieval..."):
            try:
                payload = {"query": query, "mode": search_mode, "top_k": top_k}
                response = requests.post(f"{API_URL}/search", json=payload, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    st.success(f"Found {len(results)} results in {search_mode} mode.")
                    
                    for i, res in enumerate(results):
                        score = res.get('score', 0)
                        score_color = "green" if score > 0.5 else "orange"
                        with st.expander(f"#{i+1} {res.get('id', 'Unknown')} (Score: :{score_color}[{score:.4f}])", expanded=True):
                            st.markdown(f"**Text:** {res.get('text', '')}")
                            st.markdown(f"**Reason:** `{res.get('reason', 'N/A')}`")
                            if res.get("metadata"):
                                st.json(res["metadata"])
                else:
                    st.error(f"Backend Error: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")

# ==========================================
# TAB 2: INGESTION
# ==========================================
with tab2:
    st.subheader("Teach the Database")
    text_input = st.text_area("Paste Raw Text / Article:", height=150, 
                              placeholder="Paste a Wikipedia article or document content here...")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ingest_btn = st.button("Ingest Data", use_container_width=True)
    
    if ingest_btn and text_input:
        with st.spinner("Processing NLP Pipeline..."):
            try:
                payload = {"text": text_input, "metadata": {"source": "user_input"}}
                response = requests.post(f"{API_URL}/ingest", json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    st.balloons()
                    st.success("Ingestion Complete!")
                    m1, m2 = st.columns(2)
                    m1.metric("Nodes Created", data.get('nodes_count', 0))
                    m2.metric("Edges Created", data.get('edges_count', 0))
                else:
                    st.error(f"Ingestion Failed: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# ==========================================
# TAB 3: VISUALIZATION (Centered & Spacious)
# ==========================================
with tab3:
    col_head, col_act = st.columns([6, 1])
    with col_head:
        st.subheader("Knowledge Graph Topology")
    with col_act:
        refresh = st.button("🔄 Refresh")

    # --- LAYOUT LOGIC ---
    # [1, 10, 1] creates a Ratio of 1:10:1. 
    # The middle column gets ~83% of the width.
    # The side columns act as margins.
    margin_left, graph_col, margin_right = st.columns([1, 10, 1])

    with graph_col:
        # Add some vertical spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        try:
            with st.spinner("Fetching Graph Topology..."):
                response = requests.get(f"{API_URL}/graph", timeout=5)
                
                if response.status_code == 200:
                    graph_data = response.json()
                    # Pass the data to the dedicated visualizer module
                    visualizer.render_graph(graph_data)
                else:
                    st.error(f"Failed to fetch graph data (Status: {response.status_code})")
                    
        except Exception as e:
            st.warning(f"Could not connect to visualization backend: {e}")