# src/__init__.py

# Expose config variables directly
from .config import (
    TARGET_BRANDS,
    GENERIC_CLASSES,
    ALL_CLASSES,
    DETECTION_CONF_THRESHOLD,
    NMS_IOU_THRESHOLD,
    VISUAL_MATCH_THRESHOLD
)

# Expose Utility classes
from .utils import NumpyEncoder, save_json

# to version the package
__version__ = "1.0.0"