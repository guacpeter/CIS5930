import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import compute_edge_metrics, aggregate_metrics

def main():
    base_dir = Path(__file__).parent.parent
    rgb_dir = base_dir / 'data' / 'rgb'
    gt_dir = base_dir / 'data' / 'edges'
    kp_dir = base_dir / 'data' / 'keypoints'
    
    rgb_files = sorted(list(rgb_dir.glob('*.jpg')))
    
    print(f"Evaluating Naive Baseline (Disks) on {len(rgb_files)} images...")
    
    metrics_list = []
    
    # Naive Parameter: Disk Radius
    # Edges are thin, but keypoints are sparse. 
    # If we draw small disks, precision might be high-ish, recall low.
    # If we draw large disks, recall high, precision low.
    # Let's pick a reasonable radius, e.g., 3 pixels.
    RADIUS = 3 
    
    for i, rgb_path in enumerate(rgb_files):
        img_name = rgb_path.stem
        gt_path = gt_dir / f"{img_name}.png"
        kp_path = kp_dir / f"{img_name}_keypoints.npz"
        
        if not gt_path.exists() or not kp_path.exists():
            continue
            
        # Load GT
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        h, w = gt_img.shape
        
        # Load Keypoints
        with np.load(kp_path) as data:
            keypoints = data['keypoints']
            
        # Generate Naive Edge Map (Draw Disks)
        pred_edges = np.zeros((h, w), dtype=np.uint8)
        for kp in keypoints:
            cv2.circle(pred_edges, (int(kp[0]), int(kp[1])), RADIUS, 255, -1)
            
        # Compute Metrics
        m = compute_edge_metrics(pred_edges, gt_img)
        metrics_list.append(m)
        
        if (i+1) % 50 == 0:
            print(f"[{i+1}/{len(rgb_files)}] Processed {img_name} | F1: {m['f1']:.4f}")
            
    # Aggregate
    if metrics_list:
        avg = aggregate_metrics(metrics_list)
        print("\n" + "="*35)
        print("NAIVE BASELINE RESULTS (Radius=3)")
        print("="*35)
        print(f"Precision:   {avg['precision_mean']:.4f}")
        print(f"Recall:      {avg['recall_mean']:.4f}")
        print(f"F1 Score:    {avg['f1_mean']:.4f}")
        print(f"Background:  {avg['bg_accuracy_mean']:.4f}")
        print("="*35)
    else:
        print("No images processed.")

if __name__ == "__main__":
    main()
