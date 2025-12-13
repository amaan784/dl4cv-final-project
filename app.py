import streamlit as st
import os
import pickle
from PIL import Image, ImageDraw, ImageFont
import torch
import json
import cv2
import numpy as np
import networkx as nx
import streamlit.components.v1 as components

# --- IMPORT PROJECT MODULES ---
from src.detection import ProductDetector 
from src.classification import load_clip_model, classify_visual_nn
from src.graph_builder import get_spatial_relationships, build_scene_graph, visualize_graph_html, visualize_planogram_logic
from src.config import TARGET_BRANDS

# --- 1. PAGE SETUP & STYLING ---
def setup_page():
    """Configures the Streamlit page settings and custom CSS."""
    st.set_page_config(
        page_title="Semantic and Spatial Scene Graph for Retail Bevergaes",
        page_icon="🥤",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main { background-color: #f9f9f9; }
        h1 { color: #d32f2f; }
        .stButton>button { 
            background-color: #d32f2f; 
            color: white; 
            width: 100%;
            font-size: 20px;
            padding: 10px;
        }
        div[data-testid="stExpander"] details summary p {
            font-size: 1.1rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR CONFIGURATION ---
def render_sidebar():
    """Renders the sidebar and returns configuration settings."""
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    # Model Selection
    model_type = st.sidebar.selectbox("Detection Model Architecture", ["YOLOv11", "RT-DETR"])

    if model_type == "YOLOv11":
        default_weights = os.path.join("weights", "yolov11l best model.pt") 
    else:
        default_weights = os.path.join("weights", "rtdetrl best model.pt")

    weights_path = st.sidebar.text_input("Model Weights Path", value=default_weights)

    # Thresholds
    st.sidebar.subheader("Tuning Parameters")
    conf_thresh = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.45)
    iou_thresh = st.sidebar.slider("NMS IoU Threshold", 0.1, 1.0, 0.45)
    match_thresh = st.sidebar.slider("Visual Match Threshold", 0.5, 1.0, 0.60)

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Pipeline:**
    1. **Detection and Localization:** Find objects (YOLOv11l or RT-DETR)
    2. **Recognition:** Identify brands (CLIP)
    3. **Spatial Logic:** Compute relationships (Geometry)
    4. **Graph Generation:** Structure data (NetworkX)
    """)
    
    return weights_path, model_type, conf_thresh, iou_thresh, match_thresh

# --- 3. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_pipeline_models(weights_path, model_arch):
    """
    Loads Detector, CLIP, and Anchors. Cached to prevent reloading on every click.
    """
    print(f"Loading {model_arch} from {weights_path}...")
    
    # A. Detector
    try:
        detector = ProductDetector(weights_path)
    except Exception as e:
        st.error(f"Failed to load detector weights at {weights_path}. Error: {e}")
        return None, None, None, None, None

    # B. Classifier (CLIP)
    clip_model, clip_processor = load_clip_model()
    
    # C. Visual Anchors
    anchors_path = os.path.join("weights", "visual_anchors.pkl")
    if os.path.exists(anchors_path):
        with open(anchors_path, 'rb') as f:
            data = pickle.load(f)
            ref_features = data['features'].to(clip_model.device)
            ref_labels = data['labels']
        print(f"✓ Loaded {len(ref_labels)} visual anchors.")
    else:
        st.error(f"Missing 'visual_anchors.pkl'. Please run Notebook 02.")
        return None, None, None, None, None

    return detector, clip_model, clip_processor, ref_features, ref_labels

# --- 4. VISUALIZATION HELPERS ---
def draw_detection_stage(image_pil, boxes):
    """Draws white bounding boxes for Stage 1."""
    img_det = image_pil.copy()
    draw_det = ImageDraw.Draw(img_det)
    for box in boxes:
        draw_det.rectangle(box, outline="white", width=3)
    return img_det

def draw_classification_stage(image_pil, boxes, labels, confidences):
    """Draws colored boxes and labels for Stage 2."""
    img_cls = image_pil.copy()
    draw_cls = ImageDraw.Draw(img_cls)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = box
        
        # Color Logic
        color = "#cccccc"
        if "Coca" in label: color = "#ff4d4d"      
        elif "Pepsi" in label: color = "#66b3ff"   
        elif "Fanta" in label: color = "#ffcc99"   
        elif "Sprite" in label: color = "#66ff66"  
        elif "7 Up" in label: color = "#ccffcc"    
        elif "Dew" in label: color = "#ccff33"     
        elif "UNKNOWN" in label: color = "gray"

        draw_cls.rectangle(box, outline=color, width=4)
        
        text = f"{label}"
        bbox = draw_cls.textbbox((x1, y1), text, font=font)
        draw_cls.rectangle(bbox, fill=color)
        draw_cls.text((x1, y1), text, fill="black", font=font)
        
    return img_cls

# --- 5. MAIN LOGIC ---
def main():
    setup_page()
    
    # Get Config from Sidebar
    weights_path, model_type, conf_thresh, iou_thresh, match_thresh = render_sidebar()

    # Load Models
    detector, clip_model, clip_processor, ref_features, ref_labels = load_pipeline_models(weights_path, model_type)

    # UI Header
    st.title("🥤 Retail Knowledge Graph")
    st.markdown("### Transform Shelf Images into Structured Data")

    # File Uploader
    uploaded_file = st.file_uploader("Upload a shelf image...", type=['jpg', 'jpeg', 'png', 'webp'])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        
        with st.expander("📸 Input Image", expanded=True):
            st.image(image_pil, use_container_width=True)

        # Run Button
        if st.button("Run Analysis Pipeline"):
            if detector is None:
                st.error("System not initialized. Check model paths.")
                st.stop()
                
            tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Detection (Localization)", "2️⃣ Classification (Recognition)", "3️⃣ Scene Graph (Intelligence)", "4️⃣ Planogram Logic"])
            
            # --- STAGE 1: DETECTION ---
            with tab1:
                st.info(f"Searching for objects using {model_type}...")
                with st.spinner("Running Object Detection..."):
                    det_result = detector.detect(image_pil, conf=conf_thresh, iou=iou_thresh)
                    boxes = det_result['boxes']
                    crops = det_result['crops']
                    
                    img_det = draw_detection_stage(image_pil, boxes)
                    st.image(img_det, caption=f"Localized {len(boxes)} potential items.", use_container_width=True)

            # --- STAGE 2: CLASSIFICATION ---
            with tab2:
                st.info("Identifying brands using CLIP Visual Anchors...")
                
                labels = []
                confidences = []
                prog_bar = st.progress(0)
                
                for i, crop in enumerate(crops):
                    label, score = classify_visual_nn(
                        crop, clip_model, clip_processor, ref_features, ref_labels, threshold=match_thresh
                    )
                    labels.append(label)
                    confidences.append(score)
                    prog_bar.progress((i + 1) / len(crops))
                
                prog_bar.empty()
                
                img_cls = draw_classification_stage(image_pil, boxes, labels, confidences)
                st.image(img_cls, caption="Brand Recognition Complete", use_container_width=True)
                
                counts = {}
                for l in labels: counts[l] = counts.get(l, 0) + 1
                st.metric("Detection Count", len(boxes))
                st.json(counts)

            # --- STAGE 3: GRAPH ---
            with tab3:
                st.info("Constructing Spatial Relationships...")
                with st.spinner("Building Graph..."):
                    w, h = image_pil.size
                    rels = get_spatial_relationships(boxes, labels, w, h)
                    G = build_scene_graph(boxes, labels, confidences, rels, TARGET_BRANDS)
                    
                    graph_html_path = "temp_graph.html"
                    visualize_graph_html(G, graph_html_path)
                    
                    with open(graph_html_path, 'r', encoding='utf-8') as f:
                        source_code = f.read() 
                    
                    components.html(source_code, height=750, scrolling=True)
                    st.success("Analysis Complete.")
            
            # --- STAGE 4: PLANOGRAM LOGIC ---
            with tab4:
                st.info("Visualizing Shelf Logic Flow (Left-to-Right)...")
                with st.spinner("Generating Planogram Flowchart..."):

                    flow_html_path = "temp_flow.html"
                    visualize_planogram_logic(G, flow_html_path)
                    
                    with open(flow_html_path, 'r', encoding='utf-8') as f:
                        flow_code = f.read() 
                    
                    components.html(flow_code, height=750, scrolling=True)
                    st.success("Planogram Logic Generated.")

if __name__ == "__main__":
    main()