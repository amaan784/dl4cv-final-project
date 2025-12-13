import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

def print_best_hyperparameters(tune_dir):
    """Prints best_hyperparameters.yaml from the specific experiment folder."""
    yaml_path = os.path.join(tune_dir, "best_hyperparameters.yaml")
    if os.path.exists(yaml_path):
        print("="*40)
        print(f"BEST HYPERPARAMETERS")
        print("="*40)
        with open(yaml_path, 'r') as f:
            print(f.read())
        print("="*40)
    else:
        print(f"best_hyperparameters.yaml not found in {tune_dir}")

def show_tuning_plots(tune_dir):
    """Displays the summary plots (fitness, scatter) from the experiment folder."""
    summary_images = ["tune_fitness.png", "tune_scatter_plots.png"]
    found = False
    print(f"Global Tuning Plots:")
    for img_name in summary_images:
        img_path = os.path.join(tune_dir, img_name)
        if os.path.exists(img_path):
            print(f"--- {img_name} ---")
            display(Image(filename=img_path))
            found = True
    if not found:
        print("No summary plots found.")

def print_tuning_csv(tune_dir, rows=5):
    """Prints the top rows of the summary CSV."""
    csv_path = os.path.join(tune_dir, "tune_results.csv")
    if os.path.exists(csv_path):
        try:
            print(f"\nTuning Results CSV (Top {rows} runs):")
            df = pd.read_csv(csv_path)
            display(df.head(rows))
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print("tune_results.csv not found.")

def compare_tuning_runs(root_dir, metric='metrics/mAP50(B)'):
    """
    Scans the ROOT directory (parent) for 'train', 'train2', etc.
    Plots a bar chart comparing their best mAP.
    """
    # Look for any folder starting with 'train' that has a results.csv
    # This handles the sibling structure: runs/tune/train, runs/tune/train2
    print(f"\nScanning for training runs in: {root_dir}")
    
    # Find all results.csv files recursively
    csv_files = glob.glob(os.path.join(root_dir, "train*/results.csv"))
    
    results = []
    if not csv_files:
        print("No individual 'train' folders found.")
        return

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            if metric in df.columns:
                best_score = df[metric].max()
                # Get folder name (e.g., 'train', 'train2')
                trial_name = os.path.basename(os.path.dirname(f))
                results.append((trial_name, best_score))
        except:
            pass

    if results:
        # Sort by trial name to keep order (train, train2, train3...)
        results.sort(key=lambda x: x[0])
        names, scores = zip(*results)

        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, scores, color='#1f77b4')
        plt.ylabel('Best mAP@50')
        plt.title('Comparison of Individual Tuning Iterations')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.3f}", 
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.show()
    else:
        print("Could not extract metrics from training folders.")