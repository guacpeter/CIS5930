#!/usr/bin/env python3
"""
Visualize SOLD2 edge detection results on dataset images.
Compares: Original Image | SOLD2 Detection | Ground Truth
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
import cv2
from pathlib import Path
import torch

from model.sold2_predictor import SOLD2Predictor
from model.predictor import EdgePredictor
from model.hybrid_predictor import HybridPredictor

def create_visualization(image, sold2_edges, edge_pred_edges, ground_truth, hybrid_edges=None):
    """Create a side-by-side visualization."""
    H, W = image.shape[:2]
    
    # Convert grayscale to RGB for colored overlay
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image.copy()
    
    # Create colored overlays
    # SOLD2 edges in green
    sold2_overlay = image_rgb.copy()
    sold2_overlay[sold2_edges > 0] = [0, 255, 0]
    
    # EdgePredictor edges in blue
    edge_pred_overlay = image_rgb.copy()
    edge_pred_overlay[edge_pred_edges > 0] = [255, 0, 0]
    
    # Ground truth in red
    gt_overlay = image_rgb.copy()
    if ground_truth is not None:
        gt_overlay[ground_truth > 0] = [0, 0, 255]
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_rgb, 'Original', (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(sold2_overlay, 'SOLD2 (Green)', (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(edge_pred_overlay, 'EdgePredictor (Blue)', (10, 30), font, 0.8, (255, 0, 0), 2)
    cv2.putText(gt_overlay, 'Ground Truth (Red)', (10, 30), font, 0.8, (0, 0, 255), 2)
    
    # Stack images
    top_row = np.hstack([image_rgb, sold2_overlay])
    bottom_row = np.hstack([edge_pred_overlay, gt_overlay])
    
    if hybrid_edges is not None:
        # Add hybrid overlay
        hybrid_overlay = image_rgb.copy()
        hybrid_overlay[hybrid_edges > 0] = [0, 255, 255]  # Cyan
        cv2.putText(hybrid_overlay, 'Hybrid (Cyan)', (10, 30), font, 0.8, (0, 255, 255), 2)
        # Make a 3-column layout
        top_row = np.hstack([image_rgb, sold2_overlay, edge_pred_overlay])
        bottom_row = np.hstack([hybrid_overlay, gt_overlay, np.zeros_like(image_rgb)])
    
    combined = np.vstack([top_row, bottom_row])
    return combined


def main():
    print("=" * 60)
    print("SOLD2 Edge Detection Visualization")
    print("=" * 60)
    
    # Paths
    data_dir = Path('data')
    rgb_dir = data_dir / 'rgb'
    edge_dir = data_dir / 'edges'
    kp_dir = data_dir / 'keypoints'
    output_dir = Path('results/sold2/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize predictors
    print("\nInitializing predictors...")
    
    # Use lightweight model (base_channels=16, descriptor_dim=32) to match trained checkpoint
    # Higher threshold since model only had synthetic pre-training
    sold2_pred = SOLD2Predictor(
        checkpoint_path='results/sold2/sold2_synth_best.pt',
        device='cpu',
        junction_thresh=0.3,
        line_thresh=0.5,  # Balanced threshold
        base_channels=16,  # Lightweight config
        descriptor_dim=32  # Lightweight config
    )
    edge_pred = EdgePredictor()
    hybrid_pred = HybridPredictor(
        mode='fusion',
        sold2_weight=0.5,
        sold2_checkpoint='results/sold2/sold2_synth_best.pt',
        sold2_thresh=0.5,  # Balanced threshold
        base_channels=16,   # Lightweight config
        descriptor_dim=32   # Lightweight config
    )
    
    # Select sample images
    sample_images = ['RGB_001.jpg', 'RGB_010.jpg', 'RGB_020.jpg', 'RGB_030.jpg', 'RGB_050.jpg']
    
    print(f"\nProcessing {len(sample_images)} images...")
    
    for img_name in sample_images:
        print(f"\n  Processing {img_name}...")
        
        # Load image
        img_path = rgb_dir / img_name
        if not img_path.exists():
            print(f"    Image not found: {img_path}")
            continue
        
        image = cv2.imread(str(img_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        
        # Load ground truth (same name as RGB, just .png extension)
        edge_name = img_name.replace('.jpg', '.png')
        edge_path = edge_dir / edge_name
        if edge_path.exists():
            ground_truth = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.resize(ground_truth, (W, H))
        else:
            ground_truth = None
            print(f"    Ground truth not found: {edge_path}")
        
        # Load keypoints for EdgePredictor (files have _keypoints suffix)
        kp_name = img_name.replace('.jpg', '_keypoints.npz')
        kp_path = kp_dir / kp_name
        if kp_path.exists():
            kp_data = np.load(kp_path)
            keypoints = kp_data['keypoints']
            scores = kp_data.get('scores', None)
            descriptors = kp_data.get('descriptors', None)
        else:
            keypoints = np.array([]).reshape(0, 2)
            scores = None
            descriptors = None
        
        # Run SOLD2
        sold2_edges = sold2_pred.detect_edge_map(gray)
        sold2_edges = cv2.resize(sold2_edges, (W, H))
        print(f"    SOLD2 edges: {np.count_nonzero(sold2_edges)} pixels")
        
        # Run EdgePredictor (original algorithm)
        edge_pred_edges = edge_pred.predict(keypoints, (H, W), scores, descriptors)
        print(f"    EdgePredictor edges: {np.count_nonzero(edge_pred_edges)} pixels")
        
        # Run Hybrid
        hybrid_edges = hybrid_pred.predict(keypoints, (H, W), scores, descriptors, image=gray)
        print(f"    Hybrid edges: {np.count_nonzero(hybrid_edges)} pixels")
        
        # Create visualization
        vis = create_visualization(image, sold2_edges, edge_pred_edges, ground_truth, hybrid_edges)
        
        # Save
        output_path = output_dir / f"vis_{img_name.replace('.jpg', '.png')}"
        cv2.imwrite(str(output_path), vis)
        print(f"    Saved: {output_path}")
        
        # Also save individual outputs
        cv2.imwrite(str(output_dir / f"sold2_{img_name.replace('.jpg', '.png')}"), sold2_edges)
        cv2.imwrite(str(output_dir / f"edgepred_{img_name.replace('.jpg', '.png')}"), edge_pred_edges)
        cv2.imwrite(str(output_dir / f"hybrid_{img_name.replace('.jpg', '.png')}"), hybrid_edges)
    
    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 60)
    
    # List output files
    print("\nOutput files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
