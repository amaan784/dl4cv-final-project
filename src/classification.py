import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from src.config import VISUAL_MATCH_THRESHOLD
from src.config import GENERIC_CLASSES 

def get_device():
    """Returns the appropriate device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    """
    Loads the CLIP model and processor.
    """
    print(f"Loading CLIP model: {model_name}...")
    device = get_device()
    
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    
    # Set to eval mode to disable dropout/randomness for deterministic results
    model.eval()
    
    return model, processor

def load_visual_anchors(dataset_root, model, processor):
    """
    Loads ALL images from 'train' and 'valid' folders of the Roboflow dataset
    to build the visual memory bank.
    
    Args:
        dataset_root (str): Path to the downloaded Roboflow dataset.
        model: Loaded CLIP model.
        processor: Loaded CLIP processor.
        
    Returns:
        ref_features (torch.Tensor): Stacked embeddings of all anchor images.
        ref_labels (list): List of class names corresponding to the embeddings.
    """
    device = model.device
    features_list = []
    labels_list = []
    
    # We use both Train and Valid sets to create a robust memory bank
    subsets = ['train', 'valid']
    
    print(f"Building Visual Anchor Database from: {dataset_root}")
    
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset path not found: {dataset_root}")
        return None, None

    for subset in subsets:
        subset_path = os.path.join(dataset_root, subset)
        if not os.path.exists(subset_path): continue
        
        # Iterate over class folders (e.g., "Coca-cola", "Sprite", "Otro")
        for class_name in os.listdir(subset_path):
            class_dir = os.path.join(subset_path, class_name)
            if not os.path.isdir(class_dir): continue

            # --- LABEL NORMALIZATION ---
            raw = class_name.lower().strip()

            if "coca" in raw: label = "Coca-Cola"
            elif "pepsi" in raw: label = "Pepsi"
            elif "sprite" in raw: label = "Sprite"
            elif "fanta" in raw: label = "Fanta"
            elif "otro" in raw or "other" in raw: label = "Other"

            # Robust matching for new classes
            elif "7up" in raw or "7 up" in raw: label = "7 Up"
            elif "dew" in raw: label = "Mountain Dew"

            else:
                label = class_name.title()
            
            # Process images in the folder
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')): continue
                
                img_path = os.path.join(class_dir, img_name)
                try:
                    # Open and Convert
                    image = Image.open(img_path).convert('RGB')
                    
                    # Preprocess
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    
                    # Encode (Inference)
                    with torch.no_grad():
                        emb = model.get_image_features(**inputs)
                        # Normalize vector (Critical for Cosine Similarity)
                        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                    
                    features_list.append(emb)
                    labels_list.append(label)
                    
                except Exception as e:
                    # Silently skip bad images to keep logs clean
                    pass
    
    if features_list:
        # Stack all vectors into one large matrix [N, 512]
        features_matrix = torch.cat(features_list, dim=0)
        
        unique_classes = sorted(list(set(labels_list)))
        print(f"Visual Anchors Loaded: {len(features_list)} images.")
        print(f"  Classes found: {unique_classes}")
        return features_matrix, labels_list
    else:
        print("No valid images found in the dataset folder.")
        return None, None

def classify_visual_nn(image, model, processor, ref_features, ref_labels, threshold=VISUAL_MATCH_THRESHOLD):
    """
    Classifies a cropped product image using Nearest Neighbor search against visual anchors.
    
    Args:
        image (PIL.Image or numpy.ndarray): The cropped product image.
        model, processor: CLIP components.
        ref_features (Tensor): The Visual Anchor database.
        ref_labels (list): Labels for the database.
        threshold (float): Similarity score required to accept a match.
        
    Returns:
        label (str): The predicted brand or "UNKNOWN".
        score (float): The similarity score (0.0 to 1.0).
    """
    device = model.device
    
    # Ensure image is PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Encode the Input Crop
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        crop_feat = model.get_image_features(**inputs)
        # Normalize
        crop_feat = crop_feat / crop_feat.norm(p=2, dim=-1, keepdim=True)
        
        # If anchors are on CPU but model is on GPU, move them automatically
        if ref_features.device != crop_feat.device:
            ref_features = ref_features.to(crop_feat.device)

        # Compare Crop to ALL Anchors (Matrix Multiplication)
        # Shape: [1, 512] @ [N, 512].T = [1, N] scores
        similarities = (crop_feat @ ref_features.T).squeeze(0)
        
        # Find the single best visual match
        best_score, best_idx = similarities.topk(1)
        
    score = best_score.item()
    label = ref_labels[best_idx.item()]
    
    # Apply Threshold Filtering
    if score < threshold:
        return "UNKNOWN", score
    
    # Handle Negative Class explicitly
    # If the closest match is a "Negative Anchor" (Other), return UNKNOWN
    if label in GENERIC_CLASSES:
        return "UNKNOWN", score
        
    return label, score