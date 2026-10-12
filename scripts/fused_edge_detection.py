#!/usr/bin/env python3
"""
Fused Edge Detection: EdgePredictor + SOLD2.

This script combines:
1. EdgePredictor: Your geometric algorithm (DBSCAN + MST + Convex Hull)
2. SOLD2: Deep learning line detection
3. Fusion: Intelligent blending of both approaches

The output is a single edge map that leverages both geometric structure 
from keypoint connections and learned edge patterns from SOLD2.
"""

import sys
import cv2
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.predictor import EdgePredictor
from model.sold2_predictor import SOLD2Predictor
from model.keypoint_sold2_predictor import KeypointSOLD2Predictor


class FusedEdgePredictor:
    """
    Combines EdgePredictor (keypoint-based) with SOLD2 (image-based) edge detection.
    
    Fusion modes:
    - 'union': Combine all edges from both methods (OR operation)
    - 'intersection': Keep only edges detected by both methods (AND operation)
    - 'weighted': Weighted average of both edge maps
    - 'edge_guided_sold2': Use EdgePredictor as mask to filter SOLD2 output
    - 'sold2_guided_edge': Use SOLD2 to validate EdgePredictor connections
    - 'hysteresis': Use EdgePredictor as seed "strong" edges and grow them using connected SOLD2 "weak" edges.
    """
    
    def __init__(self,
                 fusion_mode: str = 'hysteresis',
                 edge_weight: float = 0.5,
                 sold2_weight: float = 0.5,
                 use_keypoint_sold2: bool = True,
                 sold2_checkpoint: str = 'results/sold2/sold2_synth_best.pt',
                 kp_sold2_checkpoint: str = 'results/sold2/kp_sold2_best.pt',
                 device: str = 'cpu'):
        """
        Args:
            fusion_mode: How to combine edge maps
        """
        self.fusion_mode = fusion_mode
        self.edge_weight = edge_weight
        self.sold2_weight = sold2_weight
        self.use_keypoint_sold2 = use_keypoint_sold2
        
        # Initialize EdgePredictor (your algorithm)
        self.edge_predictor = EdgePredictor(
            eps=20,
            min_samples=3,
            desc_thresh=0.5,
            max_dist=20,
            gap_dist=60
        )
        
        # Initialize SOLD2 with lower threshold for hysteresis growing
        line_thresh = 0.3 if fusion_mode == 'hysteresis' else 0.4
        
        if use_keypoint_sold2 and Path(kp_sold2_checkpoint).exists():
            print(f"Using Keypoint-Conditioned SOLD2 (thresh={line_thresh})")
            self.sold2 = KeypointSOLD2Predictor(
                checkpoint_path=kp_sold2_checkpoint,
                device=device,
                junction_thresh=0.2,
                line_thresh=line_thresh,
                base_channels=16,
                descriptor_dim=32
            )
        else:
            print("Using Standard SOLD2")
            self.sold2 = SOLD2Predictor(
                checkpoint_path=sold2_checkpoint,
                device=device,
                junction_thresh=0.2,
                line_thresh=line_thresh,
                base_channels=16,
                descriptor_dim=32
            )
    
    def predict(self, 
                image: np.ndarray, 
                keypoints: np.ndarray,
                scores: np.ndarray = None,
                descriptors: np.ndarray = None) -> np.ndarray:
        """Predict fused edge map."""
        H, W = image.shape[:2]
        
        if scores is None:
            scores = np.ones(len(keypoints))
        
        # 1. Run EdgePredictor (your algorithm) - The "Strong" Edges
        edge_map_geo = self.edge_predictor.predict(
            keypoints, (H, W), scores, descriptors
        )
        
        # 2. Run SOLD2 - The "Weak" Edges (Candidates for expansion)
        if self.use_keypoint_sold2 and isinstance(self.sold2, KeypointSOLD2Predictor):
            edge_map_sold2 = self.sold2.predict(
                keypoints, (H, W), scores, image=image
            )
        else:
            edge_map_sold2 = self.sold2.detect_edge_map(image)
        
        # 3. Fuse the edge maps
        fused = self._fuse(edge_map_geo, edge_map_sold2)
        
        return fused
    
    def _fuse(self, edge_geo: np.ndarray, edge_sold2: np.ndarray) -> np.ndarray:
        """Fuse two edge maps based on the selected mode."""
        
        # Ensure same size
        if edge_geo.shape != edge_sold2.shape:
            edge_sold2 = cv2.resize(edge_sold2, (edge_geo.shape[1], edge_geo.shape[0]))
        
        if self.fusion_mode == 'hysteresis':
            # 1. Start with Geometric Edges as the definitive seed
            # 2. Find connected components in SOLD2
            # 3. Keep SOLD2 components that touch Geometric Edges
            
            # Label connected components in SOLD2 output
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_sold2, connectivity=8)
            
            # Find which labels overlap with the Geometric Edges
            # Create a mask of valid seeds (dilate slightly to ensure overlap)
            seed_mask = cv2.dilate(edge_geo, np.ones((3, 3), np.uint8))
            
            # Get labels present in the seed regions
            present_labels = np.unique(labels[seed_mask > 0])
            
            # Remove background label (0)
            present_labels = present_labels[present_labels != 0]
            
            # Create mask of valid connected components
            mask = np.isin(labels, present_labels).astype(np.uint8) * 255
            
            # Combine: Original Geo Edges + Connected SOLD2 Extensions
            fused = cv2.bitwise_or(edge_geo, mask)
            
            return fused
        
        if self.fusion_mode == 'union':
            fused = np.maximum(edge_geo, edge_sold2)
            
        elif self.fusion_mode == 'intersection':
            # AND: Only keep edges detected by both
            geo_bin = (edge_geo > 127).astype(np.uint8)
            sold2_bin = (edge_sold2 > 127).astype(np.uint8)
            fused = (geo_bin & sold2_bin) * 255
            
        elif self.fusion_mode == 'weighted':
            # Weighted average
            geo_f = edge_geo.astype(np.float32) / 255.0
            sold2_f = edge_sold2.astype(np.float32) / 255.0
            weighted = self.edge_weight * geo_f + self.sold2_weight * sold2_f
            fused = (weighted > 0.5).astype(np.uint8) * 255
            
        elif self.fusion_mode == 'edge_guided_sold2':
            # Use EdgePredictor as a mask - dilate it slightly
            geo_dilated = cv2.dilate(edge_geo, np.ones((5, 5), np.uint8))
            fused = np.where(geo_dilated > 0, edge_sold2, 0).astype(np.uint8)
            
        elif self.fusion_mode == 'sold2_guided_edge':
            # Use SOLD2 to validate EdgePredictor connections
            sold2_dilated = cv2.dilate(edge_sold2, np.ones((7, 7), np.uint8))
            fused = np.where(sold2_dilated > 0, edge_geo, 0).astype(np.uint8)
            
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        
        return fused.astype(np.uint8)


def load_keypoints(kp_path):
    """Load SuperPoint keypoints from npz file."""
    data = np.load(kp_path)
    keypoints = data['keypoints']
    scores = data['scores']
    descriptors = data.get('descriptors', None)
    return keypoints, scores, descriptors


import matplotlib.pyplot as plt

def calculate_metrics(pred, gt, tolerance=2):
    """Calculate precision, recall, F1 with tolerance."""
    if gt is None:
        return 0, 0, 0
        
    # Resize if needed
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        
    # Dilate ground truth for tolerance
    kernel = np.ones((tolerance*2+1, tolerance*2+1), np.uint8)
    gt_dilated = cv2.dilate(gt, kernel)
    pred_dilated = cv2.dilate(pred, kernel)
    
    # Metrics
    tp = np.sum((pred > 0) & (gt_dilated > 0))
    fp = np.sum((pred > 0) & (gt_dilated == 0))
    fn = np.sum((gt > 0) & (pred_dilated == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def create_visualization(image, keypoints, ground_truth, predicted, img_name, precision, recall, f1):
    """Create a 2x2 grid visualization matching the requested style."""
    H, W = image.shape[:2]
    
    # 1. Original Image (Top Left)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Keypoints on Image (Top Right)
    img_kp = img_rgb.copy()
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img_kp, (x, y), 2, (0, 255, 0), -1)  # Green dots
    
    # 3. Ground Truth (Bottom Left)
    if ground_truth is not None:
        gt_vis = ground_truth.copy()
    else:
        gt_vis = np.zeros((H, W), dtype=np.uint8)
        
    # 4. Predicted Edges (Bottom Right)
    pred_vis = predicted.copy()
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Results for {img_name}\nP: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}", fontsize=14)
    
    # Top Left
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Top Right
    axes[0, 1].imshow(img_kp)
    axes[0, 1].set_title(f"SuperPoint Keypoints (n={len(keypoints)})")
    axes[0, 1].axis('off')
    
    # Bottom Left
    axes[1, 0].imshow(gt_vis, cmap='gray')
    axes[1, 0].set_title("Ground Truth Edges")
    axes[1, 0].axis('off')
    
    # Bottom Right
    axes[1, 1].imshow(pred_vis, cmap='gray')
    axes[1, 1].set_title("Predicted Edges (Fused)")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Fused Edge Detection: EdgePredictor + SOLD2")
    print("=" * 60)
    
    # Paths
    image_dir = Path('data/rgb')
    kp_dir = Path('data/keypoints')
    edge_dir = Path('data/edges')
    output_dir = Path('results/fused_edges')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    # Initialize fused predictor
    print(f"\nInitializing FusedEdgePredictor on {device}...")
    fused_predictor = FusedEdgePredictor(
        fusion_mode='hysteresis',
        edge_weight=0.5,
        sold2_weight=0.5,
        use_keypoint_sold2=True,
        device=device
    )
    
    # Process all images in directory
    sample_images = sorted([p.name for p in image_dir.glob('*.jpg')])
    
    print(f"\nProcessing {len(sample_images)} images...")
    
    metrics = {'p': [], 'r': [], 'f1': []}
    
    for img_name in sample_images:
        img_path = image_dir / img_name
        if not img_path.exists():
            print(f"  Skipping {img_name}: not found")
            continue
        
        print(f"\n  Processing {img_name}...")
        
        # Load image
        image = cv2.imread(str(img_path))
        H, W = image.shape[:2]
        
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
        
        # Generate fused edge map
        fused_edge = fused_predictor.predict(image, keypoints, scores, descriptors)
        
        # Calculate metrics
        p, r, f1 = calculate_metrics(fused_edge, ground_truth, tolerance=3)
        print(f"    P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")
        
        # Save metrics
        metrics['p'].append(p)
        metrics['r'].append(r)
        metrics['f1'].append(f1)
        
        # Save raw output
        fused_output_path = output_dir / f"fused_{img_name.replace('.jpg', '.png')}"
        cv2.imwrite(str(fused_output_path), fused_edge)
        
        # Create styled visualization
        fig = create_visualization(
            image, keypoints, ground_truth, fused_edge, 
            img_name, p, r, f1
        )
        
        # Save visualization
        # Note: Replace 'comparison_' with 'styled_' to distinguish
        vis_path = output_dir / f"styled_{img_name.replace('.jpg', '.png')}"
        fig.savefig(str(vis_path), dpi=150)
        plt.close(fig)
        print(f"    Styled visualization saved: {vis_path}")
    
    # Calculate and print averages
    print("\n" + "=" * 60)
    if metrics['f1']:
        avg_p = np.mean(metrics['p'])
        avg_r = np.mean(metrics['r'])
        avg_f1 = np.mean(metrics['f1'])
        print(f"AVERAGE METRICS over {len(metrics['f1'])} images:")
        print(f"Precision: {avg_p:.4f}")
        print(f"Recall:    {avg_r:.4f}")
        print(f"F1 Score:  {avg_f1:.4f}")
    else:
        print("No images processed successfully.")
        
    print("=" * 60)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
