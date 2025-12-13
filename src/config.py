# src/config.py

# CLASS DEFINITIONS
# The specific brands we want to detect and highlight in the graph
TARGET_BRANDS = [
    "Coca-Cola", 
    "Sprite", 
    "Fanta", 
    "Pepsi",
    "7 Up",
    "Mountain Dew"
]

# Generic classes to catch background noise or non-target items
# not using for now
GENERIC_CLASSES = [ 
    "Other", 
    "Empty Shelf", 
    "Price Tag"
]

# Combined list for CLIP
ALL_CLASSES = TARGET_BRANDS

# THRESHOLDS
# YOLO Detection Confidence (Lower = detect more, higher = strictly confident)
DETECTION_CONF_THRESHOLD = 0.25

# NMS IoU Threshold (Lower = aggressively merge overlapping boxes)
NMS_IOU_THRESHOLD = 0.45

# CLIP Visual Matching Threshold (0.0 to 1.0)
# If a visual match score is below this, the item is labeled "UNKNOWN"
VISUAL_MATCH_THRESHOLD = 0.75

# GRAPH SETTINGS
# Distance thresholds (relative to image size) for determining relationships
HORIZONTAL_THRESHOLD_RATIO = 0.30  # 30% of image width
VERTICAL_THRESHOLD_RATIO = 0.10    # 10% of image height
SAME_ROW_THRESHOLD_RATIO = 0.20    # 20% of image height (handling slanted shelves)

# PATH
# Standard paths for your project structure
DEFAULT_WEIGHTS_PATH = "../weights/yolov11l best model.pt"
ROBOFLOW_DATASET_PATH = "../data/roboflow_refrescos"