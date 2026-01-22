#!/usr/bin/env python3
"""
Test the model, output comparison images, and compute pixel-level statistics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from model.predictor import EdgePredictor
from model.utils import load_keypoints, filter_keypoints, load_image, load_edge_gt, get_image_paths


def compute_pixel_stats(pred_edge, gt_edge):
    """Compute pixel-level statistics."""
    pred = (pred_edge > 127).astype(np.uint8)
    gt = (gt_edge > 127).astype(np.uint8)

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    tn = np.sum((pred == 0) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    return {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)}


def compute_derived_metrics(stats):
    """Compute precision, recall, F1, and background accuracy."""
    tp, fp, tn, fn = stats['TP'], stats['FP'], stats['TN'], stats['FN']

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    bg_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'bg_accuracy': bg_accuracy}


def main():
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data'
    output_dir = data_dir / 'test_output'
    output_dir.mkdir(exist_ok=True)

    paths = get_image_paths(data_dir)

    # Select sample images for visualization
    sample_indices = [0, 50, 100, 150, 200] if len(paths) > 200 else list(range(min(5, len(paths))))
    sample_paths = [paths[i] for i in sample_indices if i < len(paths)]

    print("=" * 100)
    print("EDGE DETECTION TEST - K-Means + Alpha Shape (Concave Hull)")
    print("=" * 100)

    predictor = EdgePredictor(k_range=(2, 10), alpha=0.02, max_edge_length=50)

    # Aggregate stats
    aggregate_stats = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    print(f"\nProcessing {len(paths)} images...")
    print("-" * 100)

    for i, (rgb_path, edge_path, kp_path) in enumerate(paths):
        img = load_image(rgb_path)
        h, w = img.shape[:2]

        gt_edge = load_edge_gt(edge_path, target_size=(w, h))

        keypoints, scores, _ = load_keypoints(kp_path)
        filtered_kp = filter_keypoints(keypoints, scores)

        pred_edge = predictor.predict(filtered_kp, (h, w), method='kmeans_alpha')
        stats = compute_pixel_stats(pred_edge, gt_edge)

        for key in ['TP', 'FP', 'TN', 'FN']:
            aggregate_stats[key] += stats[key]

        # Save comparison images for sample images
        if (rgb_path, edge_path, kp_path) in sample_paths:
            # Create comparison image: Original | Predicted | Ground Truth
            comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
            comparison[:, :w] = img
            comparison[:, w:2*w] = cv2.cvtColor(pred_edge, cv2.COLOR_GRAY2BGR)
            comparison[:, 2*w:] = cv2.cvtColor(gt_edge, cv2.COLOR_GRAY2BGR)

            # Add labels
            cv2.putText(comparison, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(comparison, "Predicted (kmeans_alpha)", (w + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(comparison, "Ground Truth", (2*w + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            out_path = output_dir / f"{rgb_path.stem}_kmeans_alpha_comparison.jpg"
            cv2.imwrite(str(out_path), comparison)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(paths)} images...")

    # Print results
    print("\n" + "=" * 100)
    print("PIXEL-LEVEL STATISTICS (Aggregate)")
    print("=" * 100)
    print(f"\n{'Metric':<20} {'Value':>15}")
    print("-" * 40)
    print(f"{'True Positives (TP)':<20} {aggregate_stats['TP']:>15,}")
    print(f"{'False Positives (FP)':<20} {aggregate_stats['FP']:>15,}")
    print(f"{'True Negatives (TN)':<20} {aggregate_stats['TN']:>15,}")
    print(f"{'False Negatives (FN)':<20} {aggregate_stats['FN']:>15,}")

    print("\n" + "=" * 100)
    print("DERIVED METRICS")
    print("=" * 100)

    m = compute_derived_metrics(aggregate_stats)
    print(f"\n{'Metric':<20} {'Value':>15}")
    print("-" * 40)
    print(f"{'Precision':<20} {m['precision']:>15.4f}")
    print(f"{'Recall':<20} {m['recall']:>15.4f}")
    print(f"{'F1 Score':<20} {m['f1']:>15.4f}")
    print(f"{'Background Accuracy':<20} {m['bg_accuracy']:>15.4f}")

    print(f"\n\nComparison images saved to: {output_dir}")
    print(f"Sample images: {[p[0].stem for p in sample_paths]}")

    # Open one comparison image
    import subprocess
    first_sample = sample_paths[0][0].stem
    subprocess.run(['open', str(output_dir / f"{first_sample}_kmeans_alpha_comparison.jpg")])


if __name__ == '__main__':
    main()
