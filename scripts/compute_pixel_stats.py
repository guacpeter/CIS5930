#!/usr/bin/env python3
"""
Compute pixel-level statistics for edge detection evaluation.

Reports: TP, FP, TN, FN for all methods.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from model.predictor import EdgePredictor
from model.utils import load_keypoints, filter_keypoints, load_image, load_edge_gt, get_image_paths


def compute_pixel_stats(pred_edge, gt_edge):
    """
    Compute pixel-level statistics.

    Args:
        pred_edge: Predicted edge map (binary or grayscale)
        gt_edge: Ground truth edge map (binary or grayscale)

    Returns:
        dict with TP, FP, TN, FN counts
    """
    pred = (pred_edge > 127).astype(np.uint8)
    gt = (gt_edge > 127).astype(np.uint8)

    # True Positives: predicted edge AND ground truth edge
    tp = np.sum((pred == 1) & (gt == 1))

    # False Positives: predicted edge BUT NOT ground truth edge
    fp = np.sum((pred == 1) & (gt == 0))

    # True Negatives: predicted background AND ground truth background
    tn = np.sum((pred == 0) & (gt == 0))

    # False Negatives: NOT predicted edge BUT IS ground truth edge
    fn = np.sum((pred == 0) & (gt == 1))

    return {
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn)
    }


def main():
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data'

    paths = get_image_paths(data_dir)
    methods = ['knn_alpha', 'gauging_beta', 'union', 'intersection']

    print("=" * 80)
    print("PIXEL-LEVEL STATISTICS")
    print("=" * 80)

    # Aggregate stats for each method
    all_stats = {m: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for m in methods}

    predictor = EdgePredictor(k=2, beta=0.5)

    for i, (rgb_path, edge_path, kp_path) in enumerate(paths):
        img = load_image(rgb_path)
        h, w = img.shape[:2]

        gt_edge = load_edge_gt(edge_path, target_size=(w, h))

        keypoints, scores, _ = load_keypoints(kp_path)
        filtered_kp = filter_keypoints(keypoints, scores)

        for method in methods:
            pred_edge = predictor.predict(filtered_kp, (h, w), method=method)
            stats = compute_pixel_stats(pred_edge, gt_edge)

            for key in ['TP', 'FP', 'TN', 'FN']:
                all_stats[method][key] += stats[key]

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(paths)} images...")

    # Print results
    print(f"\nProcessed {len(paths)} images total")
    print("\n" + "=" * 80)
    print("RESULTS BY METHOD")
    print("=" * 80)

    for method in methods:
        stats = all_stats[method]
        total_pixels = stats['TP'] + stats['FP'] + stats['TN'] + stats['FN']

        # Compute derived metrics
        precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
        recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0
        accuracy = (stats['TP'] + stats['TN']) / total_pixels if total_pixels > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{method.upper()}")
        print("-" * 40)
        print(f"  True Positives (TP):  {stats['TP']:>12,}")
        print(f"  False Positives (FP): {stats['FP']:>12,}")
        print(f"  True Negatives (TN):  {stats['TN']:>12,}")
        print(f"  False Negatives (FN): {stats['FN']:>12,}")
        print(f"  {'':>24} {'-'*12}")
        print(f"  Total Pixels:         {total_pixels:>12,}")
        print()
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  F1 Score:   {f1:.4f}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Method':<15} {'TP':>12} {'FP':>12} {'TN':>15} {'FN':>12}")
    print("-" * 70)
    for method in methods:
        s = all_stats[method]
        print(f"{method:<15} {s['TP']:>12,} {s['FP']:>12,} {s['TN']:>15,} {s['FN']:>12,}")


if __name__ == '__main__':
    main()
