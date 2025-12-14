# dl4cv-final-project
Final Project for COMS 4995 

# Spatial and Semantic Scene Graph: Beverage

A Computer Vision pipeline that detects retail products **(using RT-DETR / YOLO)**, classifies them using **Visual Anchors (CLIP)**, and generates a **Spatial and Semantic Scene Graph** to understand product placement (e.g., "Coke is Next To Pepsi").

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

### Datasets and Weights

Download them from this folder: https://drive.google.com/drive/folders/1ZdQjYFMLYUOi4Ke2uEdluPrw5M1Lx8Zv?usp=sharing
and just paste in the weights and data folder in the root directory of this repo

Create a 'weights' and 'data' folder if they do not exist.

### Environment
Clone the repo and install dependencies:

`pip install -r requirements.txt`

## Run stramlit -
`python -m streamlit run app.py`
 
The UI will open up. Select the model and the configurations on the left panel. Input an image of beverages on retail shelf and click "Run Analysis Pipeline".

# Tuning and Training-
1) Run datasetup notebook for downloading SKU110K (Notebook 00)

2) Run tuning notebooks (Notebooks 03, 04, 05)

3) Run final training notebook (01 based on results from step 2)

## Sample yaml file that gets created after running notebook 00-
(ideally delete it in a new environment)
names:
  0: product
path: /workspace/dl4cv-final-project/data/datasets/thedatasith/sku110k-annotations/versions/14/SKU110K_fixed
test: images/test
train: images/train
val: images/val

# CLIP few shot recognition-

Run notebook 2 

# Main pipeline testing and evaluation-

Run notebooks 06 and then run 07 for the main or overall pipeline / app testing 


# Directory-
```bash
retail_scene_graph/
├── runs/
│   ├── tune/
│   │   ├── yolo_nano_tuning/       # The Tuning Experiment
│   │   │   ├── tune_scatter_plots.png
│   │   │   ├── best_hyperparameters.yaml
│   │   │   └── weights/            # Checkpoints for the best tuned model
│   │   ├── rtdetr_tuning/
│   │   └── ...
├── data/                        # ALL input data goes here
│   ├── datasets/                
│   ├── roboflow_refrescos/      # The Classification dataset (Visual Anchors)
│   │   ├── train/               # Used for Visual Memory
│   │   ├── valid/               # Used for Visual Memory
│   │   └── test/                # Used for Golden Set Evaluation
│   └── test_images/             # Your 5-10 specific shelf images for the demo
│
├── weights/                         # Trained Model Artifacts
│   ├── rtdetr best model.pt         # Best fine-tuned RT-DETR checkpoint
│   ├── rtdetr initial finetuning.pt # Initial RT-DETR fine-tuning checkpoint
│   ├── yolov11l best model.pt       # Best YOLOv11l checkpoint
│   ├── yolov11l initial finetuning.pt # Initial YOLOv11l fine-tuning checkpoint
│   ├── visual_anchors.pkl           # Visual anchor embeddings
│   └── old_visual_anchors.pkl       # Previous version of anchor embeddings
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
├── notebooks/                   # Experiments
│   ├── 00_Data_Setup.ipynb            # Datasetup for SKU110-K
│   ├── 01_Detection_Training.ipynb    # Training RT-DETR / YOLO
│   ├── 02_CLIP_Visual_Anchors.ipynb   # CLIP few shot learning
│   ├── 03_Tune_YOLO_Nano.ipynb        # Hyperparameter Tuning for YOLO Nano
│   ├── 04_Tune_YOLO_Large.ipynb       # Hyperparameter Tuning for YOLO Large
│   └── 05_Tune_RTDETR.ipynb           # Hyperparameter Tuning for RT-DETR
│   ├── 06_Pipeline_Logic.ipynb        # Developing the pipeline (Detect -> Classify -> Graph)
│   ├── 07_Final_Evaluation.ipynb      # Generating Metrics & Report Plots

├── app.py                       # The Streamlit Dashboard
├── requirements.txt             # List of libraries (ultralytics, pyvis, etc.)
└── README.md                    # Project documentation