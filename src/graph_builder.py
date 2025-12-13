import networkx as nx
from pyvis.network import Network
from IPython.display import display, HTML
import os
from src.config import HORIZONTAL_THRESHOLD_RATIO, VERTICAL_THRESHOLD_RATIO, SAME_ROW_THRESHOLD_RATIO,TARGET_BRANDS

def get_spatial_relationships(boxes, labels, image_width, image_height):
    """
    Determine spatial relationships.
    - 'next_to': All close neighbors (Undirected / Clustering)
    - 'left_of': ONLY the single nearest neighbor to the right (Directed / Flowchart)
    """
    relationships = []
    num_products = len(boxes)
    
    horizontal_thresh = image_width * HORIZONTAL_THRESHOLD_RATIO
    vertical_thresh = image_height * VERTICAL_THRESHOLD_RATIO
    row_thresh = image_height * SAME_ROW_THRESHOLD_RATIO
    
    # Pre-calculate centers for boxes
    centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

    for i in range(num_products):
        cx_i, cy_i = centers[i]
        
        # Track the single best neighbor to the right
        nearest_right_idx = -1
        min_right_dist = float('inf')

        for j in range(num_products):
            if i == j: continue
            
            cx_j, cy_j = centers[j]
            
            # Row Alignment Check
            if abs(cy_i - cy_j) < row_thresh:
                dist_x = cx_j - cx_i
                
                # Undirected 'next_to' (Cluster everything close)
                # i < j avoids duplicates for undirected edges
                if i < j and abs(dist_x) < horizontal_thresh:
                    relationships.append((i, 'next_to', j))
                
                # Find candidates for 'left_of'
                # Check if j is to the right of i, and within range
                if 0 < dist_x < horizontal_thresh:
                    # Update nearest neighbor if this one is closer
                    if dist_x < min_right_dist:
                        min_right_dist = dist_x
                        nearest_right_idx = j

            # Vertical Check
            elif i < j:
                if cy_i < cy_j - vertical_thresh:
                    relationships.append((i, 'above', j))
                elif cy_j < cy_i - vertical_thresh:
                    relationships.append((j, 'above', i))

        # Add ONLY the single best 'left_of' connection after checking all j
        if nearest_right_idx != -1:
            relationships.append((i, 'left_of', nearest_right_idx))
            
    return relationships

def build_scene_graph(boxes, labels, confidences, relationships, target_brands):
    """
    Constructs a NetworkX graph combining Physical Items and Brand Concepts
    """
    G = nx.DiGraph()
    
    # Add Nodes (Physical Items)
    for i, (label, conf) in enumerate(zip(labels, confidences)):
        item_id = f"Item_{i}"
        G.add_node(item_id, 
                   label=label, 
                   confidence=conf,
                   box=boxes[i],
                   type="Physical_Object")
        
        # Knowledge Graph Edge
        if label in target_brands:
            if not G.has_node(label):
                G.add_node(label, type="Brand_Concept", label=label)
            G.add_edge(item_id, label, relationship="is_brand")
            
    # Add Spatial Edges
    for rel in relationships:
        idx1, type_name, idx2 = rel
        node1 = f"Item_{idx1}"
        node2 = f"Item_{idx2}"
        G.add_edge(node1, node2, relationship=type_name)
        
    return G

# --- VISUALIZATION 1: CLEAN STAR CHART (No spatial lines) ---
def visualize_polished_static(G, output_file="../results/graphs/polished_graph.html"):
    """
    Generates a clean 'Star Chart' style graph showing only Brand clusters.
    Good for high-level overviews.
    """
    directory = os.path.dirname(output_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", notebook=True, cdn_resources='in_line')
    net.toggle_physics(False)
    
    # Use Config list
    target_brands = TARGET_BRANDS

    added_brands = set()
    
    # Pre-calculate layout
    pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42, scale=500)
    
    for node, attr in G.nodes(data=True):
        label = attr.get('label', '')
        x, y = pos.get(node, (0,0))
        
        if label in target_brands and attr.get('type') != 'Brand_Concept':
            # Brand Hub
            if label not in added_brands:
                bx, by = pos.get(label, (0,0))
                net.add_node(label, label=label, 
                             color={'background': '#FFD700', 'border': '#b39700'}, 
                             size=50, title="Brand Concept", 
                             x=float(bx), y=float(by), borderWidth=2,
                             font={'size': 32, 'face': 'Arial', 'strokeWidth': 2, 'strokeColor': '#ffffff'})
                added_brands.add(label)
            
            # Bottle Node
            if "Coca" in label: bg, border = "#ff4d4d", "#990000"
            elif "Sprite" in label: bg, border = "#66ff66", "#006600"
            elif "Fanta" in label: bg, border = "#ffcc99", "#cc6600"
            elif "Pepsi" in label: bg, border = "#66b3ff", "#004d99"
            elif "7 Up" in label: bg, border = "#ccffcc", "#00cc00"   # Light Green
            elif "Dew" in label: bg, border = "#ccff33", "#99cc00"    # Neon/Lime
        
            else: bg, border = "#cccccc", "#666666"
            added_brands = set()
            
            net.add_node(node, label=" ", 
                         color={'background': bg, 'border': border}, 
                         size=12, title=f"Detected: {label}", 
                         x=float(x), y=float(y), borderWidth=1)
            
            # Edge
            net.add_edge(node, label, color="#d3d3d3", width=1)

    # Force UTF-8 encoding for Windows compatibility
    html = net.generate_html()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Polished Static Graph saved: {output_file}")
    try:
        display(HTML(output_file))
    except:
        pass
    return

# --- VISUALIZATION 2: FULL GRAPH WITH LEGEND ---
def visualize_graph_html(G, output_file="../results/graphs/shelf_graph_full.html"):
    """
    Generates the full graph including 'next_to' spatial edges and a custom Legend overlay.
    """
    directory = os.path.dirname(output_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", notebook=True, cdn_resources='in_line')
    net.toggle_physics(False)
    
    added_brands = set()
    added_nodes = set()
    target_brands = TARGET_BRANDS
    
    # Calculate layout once
    pos = nx.spring_layout(G, k=0.7, iterations=200, seed=42, scale=500)
    
    # --- PASS 1: Add ALL Brand Concept Nodes First (The Hubs) ---
    for node, attr in G.nodes(data=True):
        if attr.get('type') == 'Brand_Concept':
            label = attr.get('label', node)
            bx, by = pos.get(node, (0,0))
            
            net.add_node(node, label=label, 
                         color={'background': '#FFD700', 'border': '#b39700'}, 
                         size=50, title="Brand Concept", 
                         x=float(bx), y=float(by), borderWidth=3,
                         font={'size': 32, 'face': 'Arial', 'strokeWidth': 2, 'strokeColor': '#ffffff'})
            
            added_brands.add(node)
            added_nodes.add(node)

    # --- PASS 2: Add Physical Items and Edges (The Leaves) ---
    for node, attr in G.nodes(data=True):
        if attr.get('type') == 'Physical_Object':
            label = attr.get('label', '')
            x, y = pos.get(node, (0,0))
            
            # Only visualize if it's a target brand
            if label in target_brands:
                # Color Logic
                if "Coca" in label: bg, border = "#ff4d4d", "#990000"
                elif "Sprite" in label: bg, border = "#66ff66", "#006600"
                elif "Fanta" in label: bg, border = "#ffcc99", "#cc6600"
                elif "Pepsi" in label: bg, border = "#66b3ff", "#004d99"
                elif "7 Up" in label: bg, border = "#ccffcc", "#00cc00"
                elif "Dew" in label: bg, border = "#ccff33", "#99cc00"
                else: bg, border = "#cccccc", "#666666"
                
                # Add Node
                net.add_node(node, label=label, 
                             color={'background': bg, 'border': border}, 
                             size=12, title=f"Detected: {label}", 
                             x=float(x), y=float(y), borderWidth=1)
                added_nodes.add(node)
                
                # Add Edge to Brand (SAFE NOW because we ran Pass 1)
                if label in added_brands:
                    net.add_edge(node, label, color="#d3d3d3", width=1)

    # --- Step B: Spatial Edges ---
    for u, v, data in G.edges(data=True):
        if u in added_nodes and v in added_nodes:
            rel = data.get('relationship')
            # Check for BOTH types of connections
            if rel == 'next_to' or rel == 'left_of':
                # Draw dashed line but DISABLE arrows so it looks like a spatial map
                net.add_edge(u, v, color="#666666", width=1, dashes=True, arrows={'to': {'enabled': False}})

    html = net.generate_html()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    # Inject Legend
    legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; width: 220px; 
                background-color: rgba(255, 255, 255, 0.95); padding: 15px; 
                border: 1px solid #ccc; border-radius: 8px; font-family: Arial, sans-serif; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
        <h4 style="margin: 0 0 10px 0; color: #333;">Graph Legend</h4>
        <div style="margin-bottom: 8px;">
            <span style="display: inline-block; width: 40px; border-top: 2px solid #d3d3d3; vertical-align: middle; margin-right: 10px;"></span>
            <span style="font-size: 13px;">Is Brand (Semantic)</span>
        </div>
        <div style="margin-bottom: 8px;">
            <span style="display: inline-block; width: 40px; border-top: 2px dashed #666666; vertical-align: middle; margin-right: 10px;"></span>
            <span style="font-size: 13px;">Next To (Spatial)</span>
        </div>
        <hr style="margin: 10px 0; border: 0; border-top: 1px solid #eee;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <span style="width: 18px; height: 18px; background: #FFD700; border: 2px solid #b39700; border-radius: 50%; margin-right: 10px;"></span>
            <span style="font-size: 13px; font-weight: bold;">Brand Concept</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="width: 12px; height: 12px; background: #ff4d4d; border: 1px solid #990000; border-radius: 50%; margin-right: 13px; margin-left: 3px;"></span>
            <span style="font-size: 13px;">Detected Item</span>
        </div>
    </div>
    """

    with open(output_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    final_html = html_content.replace('</body>', f'{legend_html}</body>')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print(f"Full Graph with Legend saved: {output_file}")
    try:
        display(HTML(output_file))
    except:
        pass
        
    return


# --- VISUALIZATION 3: PLANOGRAM FLOW ---
def visualize_planogram_logic(G, output_file="../results/graphs/planogram_flow.html"):
    """
    Generates the 'Box & Arrow' flow chart.
    Uses 'shape=box' to replicate the reference image style.
    """
    directory = os.path.dirname(output_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", notebook=True, cdn_resources='in_line')
    
    # Hierarchy Layout (Left-to-Right Flow)
    net.set_options("""
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 200,
          "nodeSpacing": 100,
          "treeSpacing": 150,
          "blockShifting": true,
          "edgeMinimization": true
        }
      },
      "physics": {
        "hierarchicalRepulsion": {
          "centralGravity": 0.0,
          "springLength": 100,
          "springConstant": 0.01,
          "nodeDistance": 120,
          "damping": 0.09
        },
        "solver": "hierarchicalRepulsion"
      },
      "edges": {
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "horizontal",
          "roundness": 0.4
        }
      }
    }
    """)
    
    target_brands = TARGET_BRANDS
    added_nodes = set()

    # Top Shelf items have lower Y values (0 is top)
    # Sorting ensures they are added first and render at the top
    all_nodes = sorted(G.nodes(data=True), key=lambda x: x[1].get('box', [0, 0, 0, 0])[1])
    
    # Add Physical Objects as BOXES
    for node, attr in all_nodes:
        if attr.get('type') == 'Physical_Object':
            label = attr.get('label', '')
            if label in target_brands:
                # Colors
                if "Coca" in label: bg, border = "#ff4d4d", "#990000"
                elif "Sprite" in label: bg, border = "#66ff66", "#006600"
                elif "Fanta" in label: bg, border = "#ffcc99", "#cc6600"
                elif "Pepsi" in label: bg, border = "#66b3ff", "#004d99"
                elif "7 Up" in label: bg, border = "#ccffcc", "#00cc00"
                elif "Dew" in label: bg, border = "#ccff33", "#99cc00"
                else: bg, border = "#cccccc", "#666666"

                # shape='box' gives the Flowchart look
                net.add_node(node, label=label, shape='box', 
                             color={'background': bg, 'border': border},
                             font={'color': 'white', 'size': 24, 'face': 'Arial', 'bold': True},
                             # shapeProperties={'borderRadius': 4}, # Optional: rounded corners
                             margin=10) # Padding inside the box
                added_nodes.add(node)

    # Add 'left_of' Edges as ARROWS
    for u, v, data in G.edges(data=True):
        if data.get('relationship') == 'left_of':
            if u in added_nodes and v in added_nodes:
                net.add_edge(u, v, label="Left Of", color="#333333", width=3, arrows="to", font={'size': 16, 'align': 'horizontal', 'background': 'white'})
    # Force UTF-8 encoding for Windows compatibility
    html = net.generate_html()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Planogram Flow saved: {output_file}")

    return
