#!/usr/bin/env python3
"""
Visualize Keypoint-Conditioned SOLD2 Edge Detection.

This script compares:
1. Original EdgePredictor (your algorithm)
2. Standard SOLD2 (image only)
3. Keypoint-conditioned SOLD2 (image + SuperPoint keypoints)
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.predictor import EdgePredictor
from model.sold2_predictor import SOLD2Predictor
from model.keypoint_sold2_predictor import KeypointSOLD2Predictor


def load_keypoints(kp_path):
    """Load SuperPoint keypoints from npz file."""
    data = np.load(kp_path)
    keypoints = data['keypoints']  # (N, 2) - x, y
    scores = data['scores']  # (N,)
    descriptors = data.get('descriptors', None)  # (256, N)
    return keypoints, scores, descriptors


def overlay_edges(image, edge_map, color=(0, 255, 0), alpha=0.5):
    """Overlay edge map on image with transparency."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    overlay = image.copy()
    edge_mask = edge_map > 0
    overlay[edge_mask] = color
    
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_keypoints(image, keypoints, scores=None, max_points=500):
    """Draw keypoints on image."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    output = image.copy()
    
    # Sort by score and take top points
    if scores is not None and len(scores) > max_points:
        indices = np.argsort(scores)[-max_points:]
        keypoints = keypoints[indices]
        scores = scores[indices]
    
    for i, kp in enumerate(keypoints):
        x, y = int(kp[0]), int(kp[1])
        radius = 2
        color = (255, 0, 0)  # Blue circles for keypoints
        cv2.circle(output, (x, y), radius, color, -1)
    
    return output


def main():
    print("=" * 60)
    print("Keypoint-Conditioned SOLD2 Visualization")
    print("=" * 60)
    
    # Paths
    image_dir = Path('data/rgb')
    edge_dir = Path('data/edges')
    kp_dir = Path('data/keypoints')
    output_dir = Path('results/keypoint_sold2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check directories exist
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    print("\nInitializing predictors...")
    
    # Initialize predictors
    edge_pred = EdgePredictor()
    
    # Standard SOLD2 (lightweight)
    sold2_pred = SOLD2Predictor(
        checkpoint_path='results/sold2/sold2_synth_best.pt',
        device='cpu',
        junction_thresh=0.3,
        line_thresh=0.5,
        base_channels=16,
        descriptor_dim=32
    )
    
    # Keypoint-conditioned SOLD2 (now with trained checkpoint)
    kp_sold2_pred = KeypointSOLD2Predictor(
        checkpoint_path='results/sold2/kp_sold2_best.pt',
        device='cpu',
        junction_thresh=0.3,
        line_thresh=0.3,
        base_channels=16,
        descriptor_dim=32,
        keypoint_sigma=4.0
    )
    
    # Sample images
    sample_images = ['RGB_001.jpg', 'RGB_010.jpg', 'RGB_030.jpg']
    
    print(f"\nProcessing {len(sample_images)} images...")
    
    for img_name in sample_images:
        img_path = image_dir / img_name
        if not img_path.exists():
            print(f"  Skipping {img_name}: not found")
            continue
        
        print(f"\n  Processing {img_name}...")
        
        # Load image
        image = cv2.imread(str(img_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        
        # Load keypoints
        kp_name = img_name.replace('.jpg', '_keypoints.npz')
        kp_path = kp_dir / kp_name
        if not kp_path.exists():
            print(f"    Keypoints not found: {kp_path}")
            continue
        
        keypoints, scores, descriptors = load_keypoints(kp_path)
        print(f"    Loaded {len(keypoints)} keypoints")
        
        # Load ground truth
        edge_name = img_name.replace('.jpg', '.png')
        edge_path = edge_dir / edge_name
        ground_truth = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE) if edge_path.exists() else None
        
        # Run predictors
        # 1. EdgePredictor
        if descriptors is not None:
            # descriptors are (256, N), EdgePredictor expects (256, N)
            edge_map_ep = edge_pred.predict(keypoints, (H, W), scores, descriptors)
        else:
            edge_map_ep = edge_pred.predict(keypoints, (H, W), scores)
        print(f"    EdgePredictor: {np.sum(edge_map_ep > 0)} edge pixels")
        
        # 2. Standard SOLD2 (image only)
        edge_map_sold2 = sold2_pred.detect_edge_map(image)
        print(f"    SOLD2: {np.sum(edge_map_sold2 > 0)} edge pixels")
        
        # 3. Keypoint-conditioned SOLD2
        edge_map_kp_sold2 = kp_sold2_pred.predict(keypoints, (H, W), scores, image=image)
        print(f"    KP-SOLD2: {np.sum(edge_map_kp_sold2 > 0)} edge pixels")
        
        # Get keypoint attention map for visualization
        kp_attention = kp_sold2_pred.get_keypoint_attention(image, keypoints, scores)
        kp_attention = cv2.resize((kp_attention * 255).astype(np.uint8), (W, H))
        
        # Create visualization
        # Row 1: Original, Keypoints, KP Attention Map
        # Row 2: EdgePredictor, SOLD2, KP-SOLD2
        # Row 3: Ground Truth overlay
        
        vis_h, vis_w = H // 2, W // 2
        
        # Resize for display
        img_small = cv2.resize(image, (vis_w, vis_h))
        
        # Row 1
        kp_vis = draw_keypoints(cv2.resize(image, (vis_w, vis_h)), keypoints / 2, scores)
        kp_attn_vis = cv2.applyColorMap(cv2.resize(kp_attention, (vis_w, vis_h)), cv2.COLORMAP_JET)
        
        # Row 2 - overlay edges
        ep_vis = overlay_edges(img_small, cv2.resize(edge_map_ep, (vis_w, vis_h)), (0, 0, 255))  # Red
        sold2_vis = overlay_edges(img_small, cv2.resize(edge_map_sold2, (vis_w, vis_h)), (0, 255, 0))  # Green
        kp_sold2_vis = overlay_edges(img_small, cv2.resize(edge_map_kp_sold2, (vis_w, vis_h)), (255, 255, 0))  # Cyan
        
        if ground_truth is not None:
            gt_vis = overlay_edges(img_small, cv2.resize(ground_truth, (vis_w, vis_h)), (0, 255, 255))  # Yellow
        else:
            gt_vis = np.zeros_like(img_small)
        
        # Add labels
        def add_label(img, text):
            cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return img
        
        add_label(img_small, "Original")
        add_label(kp_vis, "SuperPoint Keypoints")
        add_label(kp_attn_vis, "Keypoint Attention")
        add_label(ep_vis, "EdgePredictor (Red)")
        add_label(sold2_vis, "SOLD2 (Green)")
        add_label(kp_sold2_vis, "KP-SOLD2 (Cyan)")
        add_label(gt_vis, "Ground Truth (Yellow)")
        
        # Combine rows
        row1 = np.hstack([img_small, kp_vis, kp_attn_vis])
        row2 = np.hstack([ep_vis, sold2_vis, kp_sold2_vis])
        
        combined = np.vstack([row1, row2])
        
        # Save
        output_path = output_dir / f"kp_sold2_{img_name.replace('.jpg', '.png')}"
        cv2.imwrite(str(output_path), combined)
        print(f"    Saved: {output_path}")
        
        # Also save individual outputs
        cv2.imwrite(str(output_dir / f"kp_attention_{img_name.replace('.jpg', '.png')}"), 
                    cv2.applyColorMap(kp_attention, cv2.COLORMAP_JET))
    
    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
