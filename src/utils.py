# src/utils.py
import json
import numpy as np
import os

def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_json(data, filepath):
    """Helper to save JSON data with Numpy support."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"Saved JSON to: {filepath}")
    
def check_data_config(config_path="../sku110k_fixed.yaml"):
    """
    Verifies that the dataset config exists and prints its content.
    Raises FileNotFoundError if missing.
    """
    # Ensure we can find the file relative to where the notebook is running
    abs_path = os.path.abspath(config_path)
    
    if os.path.exists(abs_path):
        print(f"Config found: {abs_path}")
        print("-" * 20)
        with open(abs_path, 'r') as f:
            print(f.read())
        print("-" * 20)
        return abs_path
    else:
        raise FileNotFoundError(f"Config missing at {abs_path}! Run 00_Data_Setup.ipynb first.")
        
class NumpyEncoder(json.JSONEncoder):
    """
    Custom encoder for JSON serialization of NumPy types.
    Usage: json.dump(data, f, cls=NumpyEncoder)
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def calculate_metrics(all_detections):
    """
    Calculate performance metrics across all test images.
    """
    total_detections = 0
    total_classified = 0
    total_unknown = 0
    confidences = []
    products_detected = {}
    
    for detections in all_detections:
        total_detections += len(detections['boxes'])
        for label, conf in zip(detections['labels'], detections['confidences']):
            confidences.append(conf)
            if label != "UNKNOWN":
                total_classified += 1
                products_detected[label] = products_detected.get(label, 0) + 1
            else:
                total_unknown += 1
    
    metrics = {
        'total_detections': total_detections,
        'total_classified': total_classified,
        'total_unknown': total_unknown,
        'classification_rate': total_classified / total_detections if total_detections > 0 else 0,
        'average_confidence': float(np.mean(confidences)) if confidences else 0,
        'median_confidence': float(np.median(confidences)) if confidences else 0,
        'products_detected': products_detected,
        'total_relationships': sum(len(d['relationships']) for d in all_detections)
    }
    
    return metrics