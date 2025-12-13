from ultralytics import YOLO, RTDETR
import os

def run_hyperparameter_tuning(
    model_type="YOLO", 
    model_weight="yolo11n.pt", 
    data_yaml="sku110k_fixed.yaml", 
    iterations=5,
    epochs=10,
    project_dir="runs/tune",
    batch_size=16,
    workers=16 
):
    """
    Runs hyperparameter tuning using the native Ultralytics tuner.
    """
    print(f"Starting Tuning for {model_type} ({model_weight})...")
    print(f"   Target: {iterations} iterations of {epochs} epochs each.")
    
    # Initialize Model
    if model_type == "YOLO":
        model = YOLO(model_weight)
        
        # Define Custom Search Space 
        search_space = {
            "lr0": (1e-4, 1e-2),
            "lrf": (0.01, 1.0),  
            "momentum": (0.7, 0.98),
            "weight_decay": (0.0, 0.001), 
            "box": (0.02, 0.2),        
            "cls": (0.2, 4.0),         
            "hsv_h": (0.0, 0.05),      
            "mosaic": (0.0, 1.0),      
            "mixup": (0.0, 1.0),
        }
    elif model_type == "RTDETR":
        model = RTDETR(model_weight)
        
        search_space = {
            "lr0": (1e-5, 1e-3),   
            "lrf": (0.01, 1.0),
            "momentum": (0.9, 0.95),   # Stricter momentum
            "weight_decay": (0.0001, 0.0005), 
            "box": (0.02, 0.2),
            "cls": (0.5, 4.0),
            # RT-DETR often doesn't use strong mosaic/mixup
            "mosaic": (0.0, 0.5),      
            "mixup": (0.0, 0.2),
        }
    else:
        raise ValueError("Unknown model type. Choose 'YOLO' or 'RTDETR'")

    # Run Tuning
    results = model.tune(
        data=data_yaml,
        epochs=epochs,
        iterations=iterations,
        optimizer="AdamW",
        plots=True, 
        save=True, 
        val=True, 
        project=project_dir,
        name=f"tune_{model_type}_{model_weight.replace('.pt', '')}",
        space=search_space,
        
        # Hardware settings
        batch=batch_size,
        imgsz=640,
        workers=workers,
        exist_ok=True 
    )
    
    return results

if __name__ == "__main__":
    # Test run
    run_hyperparameter_tuning(
        model_type="YOLO",
        model_weight="yolo11n.pt",
        data_yaml="sku110k_fixed.yaml",
        iterations=2,
        epochs=2
    )