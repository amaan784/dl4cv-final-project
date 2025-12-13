import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO, RTDETR
from src.config import DETECTION_CONF_THRESHOLD, NMS_IOU_THRESHOLD

class ProductDetector:
    """
    Handles object detection using YOLO or RT-DETR models.
    """
    def __init__(self, model_path):
        """
        Initialize the detector.
        
        Args:
            model_path (str): Path to the .pt file (e.g., 'weights/best_rtdetr.pt')
        """
        print(f"Loading Detection Model: {model_path}...")
        
        # Auto-detect model type based on filename or try/except logic
        try:
            if "rtdetr" in model_path.lower():
                self.model = RTDETR(model_path)
                print("Detected RT-DETR architecture.")
            else:
                self.model = YOLO(model_path)
                print("Detected YOLO architecture.")
        except Exception as e:
            print(f"Warning: Could not auto-load. Defaulting to YOLO class. Error: {e}")
            self.model = YOLO(model_path)

    def detect(self, image_input, conf=DETECTION_CONF_THRESHOLD, iou=NMS_IOU_THRESHOLD):
        """
        Runs inference on an image and extracts crops.

        Args:
            image_input (str or PIL.Image): Path to image or PIL Image object.
            conf (float): Confidence threshold (default from config.py).
            iou (float): NMS IoU threshold (default from config.py).

        Returns:
            results_dict (dict): Contains 'image', 'boxes', and 'crops'.
        """
        # Ensure input is a PIL Image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        # Run Inference
        # agnostic_nms=True is CRITICAL to prevent "double boxes" on the same bottle
        results = self.model(
            image, 
            conf=conf, 
            iou=iou, 
            agnostic_nms=True,
            verbose=False
        )

        detected_boxes = []
        cropped_images = []

        # Process results
        for result in results:
            # result.boxes.xyxy returns [x1, y1, x2, y2]
            # .cpu().numpy() converts it from GPU tensor to standard array
            boxes = result.boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Sanity check for edge cases (negative coordinates)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.width, x2), min(image.height, y2)

                # Crop the object for the Classifier
                crop = image.crop((x1, y1, x2, y2))
                
                detected_boxes.append([x1, y1, x2, y2])
                cropped_images.append(crop)

        return {
            "original_image": image,
            "boxes": detected_boxes,
            "crops": cropped_images,
            "count": len(detected_boxes)
        }

# Usage Example (if run directly):
if __name__ == "__main__":
    detector = ProductDetector("weights/best_rtdetr.pt")
    # result = detector.detect("data/test_images/shelf.jpg")
    # print(f"Detected {result['count']} items.")