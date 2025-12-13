import os
import yaml
import kagglehub
from pathlib import Path

# CONFIGURATION
# Get the path of THIS file (src/data_loader.py)
SRC_DIR = Path(__file__).resolve().parent

# Go up one level to the project root, then into 'data'
DATA_DIR = (SRC_DIR.parent / "data").resolve()

print(f"Data Directory set to: {DATA_DIR}")

def load_sku110k_data(target_yaml_path="../sku110k_fixed.yaml"):
    """
    1. SEARCHES for 'SKU110K_fixed' in the data folder.
    2. If found -> Generates config immediately.
    3. If missing -> Downloads it, then generates config.
    """
    config_path = Path(target_yaml_path).resolve()
    found_data_path = None

    # SEARCH LOCALLY FIRST
    # We search recursively because the folder might be nested like:
    # data/datasets/thedatasith/.../SKU110K_fixed
    # or just data/SKU110K_fixed
    
    if DATA_DIR.exists():
        print(f"Searching for 'SKU110K_fixed' in {DATA_DIR}...")
        results = list(DATA_DIR.rglob("SKU110K_fixed"))
        if results:
            found_data_path = results[0]
            print(f"Found existing data at: {found_data_path}")

    # DOWNLOAD ONLY IF TRULY MISSING
    if not found_data_path:
        print("Data not found locally. Downloading from Kaggle...")
        
        # We explicitly tell Kaggle to download to our DATA_DIR
        # This keeps it out of hidden system folders
        os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)

        try:
            # Download
            dataset_root = kagglehub.dataset_download("thedatasith/sku110k-annotations")
            
            # Search again after download
            found_data_path = list(DATA_DIR.rglob("SKU110K_fixed"))[0]
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None

    # GENERATE YAML CONFIG
    
    sku_data_config = {
        'path': found_data_path.as_posix(),  # Absolute path to the data
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': { 0: 'product' }
    }

    with open(config_path, 'w') as f:
        yaml.dump(sku_data_config, f)

    print(f"Config saved to: {config_path}")
    return str(config_path)