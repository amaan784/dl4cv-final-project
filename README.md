# dl4cv-final-project
Final Project for COMS 4995 

# Retail Scene Graph: Beverage Detection & Analysis

A Computer Vision pipeline that detects retail products (beverages), classifies them using **Visual Anchors (CLIP)**, and generates a **Semantic Scene Graph** to understand product placement (e.g., "Coke is Next To Pepsi").

## Architecture

1.  **Detection (The "Eye"):** * Model: **RT-DETR (Large)** / YOLOv11
    * Task: Localize all bottles/cans on the shelf.
    * Training Data: SKU-110K Dataset.
2.  **Classification (The "Brain"):**
    * Model: **CLIP (ViT-B/32)**
    * Method: **Visual Anchor Matching (Few-Shot)**. We compare detected crops against a memory bank of reference images (Coca-Cola, Sprite, etc.) rather than training a new classifier.
3.  **Graph Construction (The "Logic"):**
    * Logic: Spatial algorithms calculate `next_to`, `above`, and `below` relationships.
    * Output: Interactive Knowledge Graph.

##  Setup & Installation

### Environment
Clone the repo and install dependencies:
```bash
pip install -r requirements.txt

# Tuning and Training-
1) Run datasetup notebook

2) Run tuning notebooks

3) Run final training notebook


# Sample yaml file that gets created-
(ideally delete it in a new environment)
names:
  0: product
path: /workspace/dl4cv-final-project/data/datasets/thedatasith/sku110k-annotations/versions/14/SKU110K_fixed
test: images/test
train: images/train
val: images/val


# Directory-
retail_scene_graph/
├── runs/
│   ├── tune/
│   │   ├── yolo_nano_tuning/       # The Tuning Experiment
│   │   │   ├── tune_scatter_plots.png
│   │   │   ├── best_hyperparameters.yaml  <-- THE GOLD!
│   │   │   └── weights/            # Checkpoints for the best tuned model
│   │   ├── rtdetr_tuning/
│   │   └── ...
├── data/                        # ALL input data goes here
│   ├── sku110k/                 # (Optional) Raw SKU-110K data if downloaded locally
│   ├── roboflow_refrescos/      # The Classification dataset (Visual Anchors)
│   │   ├── train/               # Used for Visual Memory
│   │   ├── valid/               # Used for Visual Memory
│   │   └── test/                # Used for Golden Set Evaluation
│   └── test_images/             # Your 5-10 specific shelf images for the demo
│
├── weights/                     # Trained Model Artifacts
│   ├── best_rtdetr.pt           # Your fine-tuned RT-DETR model
│   └── best_yolo.pt             # (Optional) Your YOLO model if you kept it
│
├── src/                         # The "Brain" (Shared Python Modules)
│   ├── __init__.py              # Makes this folder importable
│   ├── config.py                # Store lists like 'TARGET_BRANDS' here
│   ├── detection.py             # RT-DETR / YOLO inference logic
│   ├── classification.py        # CLIP Visual Anchor loading & Nearest Neighbor logic
│   ├── graph_builder.py         # NetworkX & PyVis visualization logic
│   ├── tuning.py                # Hyperparameter tuning logic (Native Ultralytics)
│   └── utils.py                 # Helpers (e.g., NumpyEncoder for JSON saving)
│
├── notebooks/                   # Your Experiments (The "Lab")
│   ├── 01_Detection_Training.ipynb    # Training RT-DETR on RunPod
│   ├── 02_Pipeline_Logic.ipynb        # Developing the pipeline (Detect -> Classify -> Graph)
│   ├── 03_Final_Evaluation.ipynb      # Generating Metrics & Report Plots
│   ├── 04_Tune_YOLO_Nano.ipynb        # Hyperparameter Tuning for YOLO Nano
│   ├── 05_Tune_YOLO_Large.ipynb       # Hyperparameter Tuning for YOLO Large
│   └── 06_Tune_RTDETR.ipynb           # Hyperparameter Tuning for RT-DETR
│
├── app.py                       # The Streamlit Dashboard
├── requirements.txt             # List of libraries (ultralytics, pyvis, etc.)
└── README.md                    # Project documentation