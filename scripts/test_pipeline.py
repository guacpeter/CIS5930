"""
Test script to run the full pipeline on a few test images and verify results.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from superpoint.extract_keypoints import KeypointExtractor
from model.predictor import EdgePredictor
from evaluation.metrics import compute_edge_metrics, aggregate_metrics

def visualize_results(rgb_path, gt_path, keypoints, pred_edges, metrics, output_path):
    """Create and save visualization of results."""
    # Read RGB image
    rgb_img = cv2.imread(str(rgb_path))
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    # Read GT edge map
    gt_edges = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Results for {rgb_path.name}\n"
                 f"P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}", 
                 fontsize=16)
    
    # 1. Original RGB
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # 2. Keypoints on Image
    kp_img = rgb_img.copy()
    for kp in keypoints:
        cv2.circle(kp_img, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
    axes[0, 1].imshow(kp_img)
    axes[0, 1].set_title(f"SuperPoint Keypoints (n={len(keypoints)})")
    axes[0, 1].axis('off')
    
    # 3. Ground Truth Edges
    axes[1, 0].imshow(gt_edges, cmap='gray')
    axes[1, 0].set_title("Ground Truth Edges")
    axes[1, 0].axis('off')
    
    # 4. Predicted Edges
    axes[1, 1].imshow(pred_edges, cmap='gray')
    axes[1, 1].set_title("Predicted Edges (k-NN)")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    rgb_dir = base_dir / 'data' / 'rgb'
    gt_dir = base_dir / 'data' / 'edges'
    output_dir = base_dir / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    print("Initializing Keypoint Extractor...")
    extractor = KeypointExtractor(cuda=False) # Use CPU for compatibility
    
    print("Initializing Edge Predictor...")
    # Combined Layout: DBSCAN (eps=20) + Descriptor Filter (thresh=0.5)
    predictor = EdgePredictor(eps=20, min_samples=3, desc_thresh=0.5, max_dist=20, gap_dist=60)
    
    # Get test images 
    # Try to find images that match, remember rgb are .jpg and edges are .png
    rgb_files = sorted(list(rgb_dir.glob('*.jpg')))  # Run on ALL images
    
    if not rgb_files:
        print("No images found in data/rgb")
        return
        
    all_metrics = []
    
    print(f"\nProcessing {len(rgb_files)} test images...")
    
    results_list = []
    
    print(f"\nProcessing {len(rgb_files)} test images...")
    
    for i, rgb_path in enumerate(rgb_files):
        img_name = rgb_path.stem # e.g. RGB_001
        gt_path = gt_dir / f"{img_name}.png"
        
        if not gt_path.exists():
            continue
            
        # 1. Extract Keypoints (Try Cache First)
        try:
            kp_path = base_dir / 'data' / 'keypoints' / f"{img_name}_keypoints.npz"
            if kp_path.exists():
                # Load from cache (FAST)
                with np.load(kp_path) as data:
                    keypoints = data['keypoints']
                    scores = data['scores']
                    descriptors = data['descriptors'].T if data['descriptors'].shape[0] == 256 else data['descriptors']
                    # Handle descriptor shape (Predictor expects 256xN, but npz might be N x 256 depending on how it was saved)
                    # Let's verify shape. Extractor usually returns 256xN.
                    
                    # Original extractor returns dict: descriptors is 256xN.
                    # extract_batch saves: descriptors=result['descriptors'] which is 256xN.
                    # BUT np.savez might save it as is.
                    # Standard assumption: Descriptors should be 256xN for Predictor.
                    
                    if descriptors.shape[0] != 256 and descriptors.shape[1] == 256:
                         descriptors = descriptors.T
            else:
                # Run Inference (SLOW)
                kp_result = extractor.extract(rgb_path)
                keypoints = kp_result['keypoints']
                scores = kp_result['scores']
                descriptors = kp_result['descriptors']
            
            # 2. Predict Edges
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            pred_edges = predictor.predict(keypoints, gt_img.shape, scores=scores, descriptors=descriptors)
            
            # 3. Compute Metrics
            metrics = compute_edge_metrics(pred_edges, gt_img)
            
            results_list.append({
                'id': img_name,
                'rgb_path': rgb_path,
                'gt_path': gt_path,
                'metrics': metrics,
                # We won't cache edges to save memory, will re-run for vis
            })
            
            # Progress print
            if (i+1) % 10 == 0:
                print(f"[{i+1}/{len(rgb_files)}] Processed {img_name} | F1: {metrics['f1']:.4f}")

        except Exception as e:
            print(f"  Error processing {img_name}: {e}")
            continue

    # --- Post-Processing ---
    if results_list:
        # 1. Save Full Stats
        stats_path = output_dir / "full_stats.md"
        with open(stats_path, "w") as f:
            f.write("# Per-Image Statistics\n\n")
            f.write("| Image ID | Precision | Recall | F1 Score | Background Accuracy |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for res in results_list:
                m = res['metrics']
                f.write(f"| **{res['id']}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['bg_accuracy']:.4f} |\n")
        print(f"\nSaved full per-image stats to {stats_path}")

        # 2. Global Averages
        avg = aggregate_metrics([r['metrics'] for r in results_list])
        print("\n" + "="*35)
        print("AVERAGE RESULTS")
        print("="*35)
        print(f"Precision:   {avg['precision_mean']:.4f} ± {avg['precision_std']:.4f}")
        print(f"Recall:      {avg['recall_mean']:.4f} ± {avg['recall_std']:.4f}")
        print(f"F1 Score:    {avg['f1_mean']:.4f} ± {avg['f1_std']:.4f}")
        print(f"Background:  {avg['bg_accuracy_mean']:.4f} ± {avg['bg_accuracy_std']:.4f}")
        print("="*35)
        
        # 3. Top 5 / Worst 5 Visualization
        # Sort by F1
        sorted_results = sorted(results_list, key=lambda x: x['metrics']['f1'], reverse=True)
        top_5 = sorted_results[:5]
        worst_5 = sorted_results[-5:]
        
        print("\nGenerating visualizations for Top 5 and Worst 5...")
        
        def process_vis(item, label, rank):
            # Re-run extraction/prediction to get visualization data
            # (We could have cached, but this matches requested "top 5" logic cleanly)
            kp_res = extractor.extract(item['rgb_path'])
            gt_img = cv2.imread(str(item['gt_path']), cv2.IMREAD_GRAYSCALE)
            pred = predictor.predict(kp_res['keypoints'], gt_img.shape, scores=kp_res['scores'], descriptors=kp_res['descriptors'])
            
            out_name = f"{label}_{rank}_{item['id']}_f1_{item['metrics']['f1']:.3f}.png"
            visualize_results(item['rgb_path'], item['gt_path'], kp_res['keypoints'], pred, item['metrics'], output_dir / out_name)
            print(f"  Saved {out_name}")

        for idx, item in enumerate(top_5):
            process_vis(item, "TOP", idx+1)
            
        for idx, item in enumerate(worst_5):
            process_vis(item, "WORST", idx+1)

        print(f"\nResults saved to {output_dir}")
    else:
        print("No metrics computed.")

if __name__ == "__main__":
    main()
