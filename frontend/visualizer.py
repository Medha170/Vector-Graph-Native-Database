import random
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# --- THEME COLORS ---
BG_COLOR = "#0E1117"
EDGE_COLOR = "#606060" 
TEXT_COLOR = "#FFFFFF"

# A palette of soft cyberpunk colors for nodes
PALETTE = [
    "#00E5FF", "#FF4B8B", "#7C4DFF", "#00FFB3", "#FFD93D",
    "#FF6F61", "#40C4FF", "#B388FF", "#69F0AE"
]

def pick_color():
    """Returns a random color from the palette."""
    return random.choice(PALETTE)

def render_graph(graph_data):
    """
    Renders a dynamic graph based on API data.
    Designed to fill the container width provided by the parent app.
    """
    if not graph_data or not graph_data.get("nodes"):
        st.info("No graph data to display yet. Try ingesting some text first!")
        return

    # --- PARSE NODES ---
    nodes = []
    existing_ids = set()
    
    # We iterate through the data from the API
    for n in graph_data.get("nodes", []):
        node_id = n.get("id", "Unknown")
        
        if node_id in existing_ids:
            continue
        existing_ids.add(node_id)

        # Assign random color and random initial position to prevent stacking
        color = pick_color()
        initial_x = random.randint(-500, 500)
        initial_y = random.randint(-500, 500)

        # Create Node Object
        nodes.append(
            Node(
                id=node_id,
                label=node_id,
                size=30,               # Large visibility
                color=color,
                shape="dot",
                x=initial_x,           # Random start pos helps physics engine
                y=initial_y,
                borderWidth=2,
                font={"color": TEXT_COLOR, "size": 14},
                title=f"Entity: {node_id}" # Tooltip
            )
        )

    # --- PARSE EDGES ---
    edges = []
    for e in graph_data.get("links", []):
        edges.append(
            Edge(
                source=e["source"],
                target=e["target"],
                label=e.get("type", ""),
                color=EDGE_COLOR,
                width=2,
                arrows="to",
                font={"color": "#CCCCCC", "size": 10, "align": "middle"},
                smooth={"type": "curvedCW", "roundness": 0.2} # Curved lines look cleaner
            )
        )

    # --- CONFIGURATION ---
    # width="100%" ensures it fills the Streamlit column (80% area defined in app.py)
    config = Config(
        width="100%",
        height="2000px",
        directed=True,
        backgroundColor=BG_COLOR,
        nodeHighlightBehavior=True,
        highlightColor="#F7A557",
        collapsible=False,
        
        # Physics Engine: Tuned for spreading nodes out
        physics={
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -100, # Repulsion
                "centralGravity": 0.005,       # Gentle pull to center
                "springLength": 200,           # Long edges
                "springConstant": 0.05,        # Bouncy
                "avoidOverlap": 1              # Prevent nodes covering each other
            },
            "stabilization": {
                "enabled": True,
                "iterations": 200,
                "fit": True
            }
        }
    )

    # Render the graph
    return agraph(nodes=nodes, edges=edges, config=config)