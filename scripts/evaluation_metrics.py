#!/usr/bin/env python3
"""
Evaluation Metrics for Edge Detection.

Reports per-image statistics and dataset-level averages for:
- True Positives (TP): Correctly predicted edge pixels
- False Positives (FP): Pixels predicted as edges but are background
- True Negatives (TN): Correctly predicted background pixels
- False Negatives (FN): Missed edge pixels

Derived metrics:
- Precision (Edge correctness): TP / (TP + FP)
- Recall (Edge coverage): TP / (TP + FN)
- F1 Score: 2 * Precision * Recall / (Precision + Recall)
- Background Accuracy: TN / (TN + FP)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
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


def compute_derived_metrics(stats):
    """Compute precision, recall, F1, and background accuracy from stats."""
    tp, fp, tn, fn = stats['TP'], stats['FP'], stats['TN'], stats['FN']

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    bg_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'bg_accuracy': bg_accuracy
    }


def main():
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data'

    paths = get_image_paths(data_dir)
    methods = ['knn_alpha', 'gauging_beta', 'union', 'intersection']

    print("=" * 100)
    print("EDGE DETECTION EVALUATION METRICS")
    print("=" * 100)
    print(f"Dataset: {len(paths)} images")
    print()

    # Store per-image and aggregate stats
    per_image_stats = {m: [] for m in methods}
    aggregate_stats = {m: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for m in methods}

    predictor = EdgePredictor(k=2, beta=0.5, alpha=0.02)

    # =========================================================================
    # PER-IMAGE STATISTICS
    # =========================================================================
    print("=" * 100)
    print("PER-IMAGE STATISTICS")
    print("=" * 100)
    print()

    for method in methods:
        print(f"\n{'='*80}")
        print(f"Method: {method.upper()}")
        print("=" * 80)
        print(f"{'Image':<12} {'TP':>10} {'FP':>10} {'TN':>12} {'FN':>10} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("-" * 80)

        for rgb_path, edge_path, kp_path in paths:
            img = load_image(rgb_path)
            h, w = img.shape[:2]

            gt_edge = load_edge_gt(edge_path, target_size=(w, h))

            keypoints, scores, _ = load_keypoints(kp_path)
            filtered_kp = filter_keypoints(keypoints, scores)

            pred_edge = predictor.predict(filtered_kp, (h, w), method=method)
            stats = compute_pixel_stats(pred_edge, gt_edge)
            metrics = compute_derived_metrics(stats)

            # Store per-image
            per_image_stats[method].append({
                'image': rgb_path.stem,
                **stats,
                **metrics
            })

            # Aggregate
            for key in ['TP', 'FP', 'TN', 'FN']:
                aggregate_stats[method][key] += stats[key]

            print(f"{rgb_path.stem:<12} {stats['TP']:>10,} {stats['FP']:>10,} {stats['TN']:>12,} {stats['FN']:>10,} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f} {metrics['f1']:>8.4f}")

        # Per-method summary
        print("-" * 80)
        agg = aggregate_stats[method]
        print(f"{'TOTAL':<12} {agg['TP']:>10,} {agg['FP']:>10,} {agg['TN']:>12,} {agg['FN']:>10,}")

    # =========================================================================
    # DATASET-LEVEL AVERAGES
    # =========================================================================
    print("\n" + "=" * 100)
    print("DATASET-LEVEL AVERAGES")
    print("=" * 100)

    for method in methods:
        stats_list = per_image_stats[method]

        avg_precision = np.mean([s['precision'] for s in stats_list])
        avg_recall = np.mean([s['recall'] for s in stats_list])
        avg_f1 = np.mean([s['f1'] for s in stats_list])
        avg_bg_acc = np.mean([s['bg_accuracy'] for s in stats_list])

        std_precision = np.std([s['precision'] for s in stats_list])
        std_recall = np.std([s['recall'] for s in stats_list])
        std_f1 = np.std([s['f1'] for s in stats_list])
        std_bg_acc = np.std([s['bg_accuracy'] for s in stats_list])

        print(f"\n{method.upper()}")
        print("-" * 50)
        print(f"  Precision (Edge correctness):  {avg_precision:.4f} +/- {std_precision:.4f}")
        print(f"  Recall (Edge coverage):        {avg_recall:.4f} +/- {std_recall:.4f}")
        print(f"  F1 Score:                      {avg_f1:.4f} +/- {std_f1:.4f}")
        print(f"  Background Accuracy:           {avg_bg_acc:.4f} +/- {std_bg_acc:.4f}")

    # =========================================================================
    # AGGREGATE STATISTICS (Total across all images)
    # =========================================================================
    print("\n" + "=" * 100)
    print("AGGREGATE STATISTICS (Total across all images)")
    print("=" * 100)
    print(f"\n{'Method':<15} {'TP':>12} {'FP':>12} {'TN':>15} {'FN':>12}")
    print("-" * 70)

    for method in methods:
        s = aggregate_stats[method]
        print(f"{method:<15} {s['TP']:>12,} {s['FP']:>12,} {s['TN']:>15,} {s['FN']:>12,}")

    # =========================================================================
    # DERIVED METRICS FROM AGGREGATE
    # =========================================================================
    print("\n" + "=" * 100)
    print("DERIVED METRICS (from aggregate counts)")
    print("=" * 100)
    print(f"\n{'Method':<15} {'Precision':>12} {'Recall':>12} {'F1':>12} {'BG Accuracy':>15}")
    print("-" * 70)

    best_f1 = 0
    best_method = None

    for method in methods:
        agg = aggregate_stats[method]
        m = compute_derived_metrics(agg)
        print(f"{method:<15} {m['precision']:>12.4f} {m['recall']:>12.4f} {m['f1']:>12.4f} {m['bg_accuracy']:>15.4f}")

        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_method = method

    print("-" * 70)
    print(f"\nBest method by F1: {best_method} (F1 = {best_f1:.4f})")

    # =========================================================================
    # TOP 5 BEST AND WORST IMAGES BY F1 SCORE
    # =========================================================================
    print("\n" + "=" * 100)
    print("TOP 5 BEST AND WORST IMAGES BY F1 SCORE")
    print("=" * 100)

    for method in methods:
        stats_list = per_image_stats[method]
        sorted_by_f1 = sorted(stats_list, key=lambda x: x['f1'], reverse=True)

        print(f"\n{method.upper()}")
        print("-" * 80)

        print("\n  TOP 5 BEST IMAGES:")
        print(f"  {'Rank':<6} {'Image':<15} {'F1':>10} {'Precision':>12} {'Recall':>10}")
        print("  " + "-" * 55)
        for i, s in enumerate(sorted_by_f1[:5], 1):
            print(f"  {i:<6} {s['image']:<15} {s['f1']:>10.4f} {s['precision']:>12.4f} {s['recall']:>10.4f}")

        print("\n  TOP 5 WORST IMAGES:")
        print(f"  {'Rank':<6} {'Image':<15} {'F1':>10} {'Precision':>12} {'Recall':>10}")
        print("  " + "-" * 55)
        for i, s in enumerate(sorted_by_f1[-5:][::-1], 1):
            print(f"  {i:<6} {s['image']:<15} {s['f1']:>10.4f} {s['precision']:>12.4f} {s['recall']:>10.4f}")

    # =========================================================================
    # EXPORT TO EXCEL
    # =========================================================================
    print("\n" + "=" * 100)
    print("EXPORTING RESULTS TO EXCEL")
    print("=" * 100)

    output_dir = script_dir / 'results'
    output_dir.mkdir(exist_ok=True)
    excel_path = output_dir / 'evaluation_metrics.xlsx'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: All per-image stats for each method
        for method in methods:
            df = pd.DataFrame(per_image_stats[method])
            df = df.rename(columns={
                'image': 'Image',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1': 'F1 Score',
                'bg_accuracy': 'BG Accuracy'
            })
            df = df[['Image', 'TP', 'FP', 'TN', 'FN', 'Precision', 'Recall', 'F1 Score', 'BG Accuracy']]
            df = df.sort_values('F1 Score', ascending=False)
            df.to_excel(writer, sheet_name=f'{method}_all', index=False)

        # Sheet 2: Top 5 best and worst for each method
        for method in methods:
            stats_list = per_image_stats[method]
            sorted_by_f1 = sorted(stats_list, key=lambda x: x['f1'], reverse=True)

            best_5 = sorted_by_f1[:5]
            worst_5 = sorted_by_f1[-5:][::-1]

            best_df = pd.DataFrame(best_5)
            best_df['Rank'] = range(1, len(best_df) + 1)
            best_df['Category'] = 'Best'

            worst_df = pd.DataFrame(worst_5)
            worst_df['Rank'] = range(1, len(worst_df) + 1)
            worst_df['Category'] = 'Worst'

            combined = pd.concat([best_df, worst_df], ignore_index=True)
            combined = combined.rename(columns={
                'image': 'Image',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1': 'F1 Score',
                'bg_accuracy': 'BG Accuracy'
            })
            combined = combined[['Category', 'Rank', 'Image', 'TP', 'FP', 'TN', 'FN', 'Precision', 'Recall', 'F1 Score', 'BG Accuracy']]
            combined.to_excel(writer, sheet_name=f'{method}_top5', index=False)

        # Sheet 3: Summary comparison across methods
        summary_data = []
        for method in methods:
            agg = aggregate_stats[method]
            m = compute_derived_metrics(agg)
            stats_list = per_image_stats[method]
            summary_data.append({
                'Method': method,
                'Avg Precision': np.mean([s['precision'] for s in stats_list]),
                'Avg Recall': np.mean([s['recall'] for s in stats_list]),
                'Avg F1': np.mean([s['f1'] for s in stats_list]),
                'Avg BG Accuracy': np.mean([s['bg_accuracy'] for s in stats_list]),
                'Agg Precision': m['precision'],
                'Agg Recall': m['recall'],
                'Agg F1': m['f1'],
                'Agg BG Accuracy': m['bg_accuracy']
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\nResults exported to: {excel_path}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print("""
Pixel-Level Statistics (Required):
  1. True Positives (TP)  - Correctly predicted edge pixels
  2. False Positives (FP) - Pixels predicted as edges but are background
  3. True Negatives (TN)  - Correctly predicted background pixels
  4. False Negatives (FN) - Missed edge pixels

Derived Metrics:
  - Precision (Edge correctness) = TP / (TP + FP)
  - Recall (Edge coverage)       = TP / (TP + FN)
  - F1 Score                     = 2 * Precision * Recall / (Precision + Recall)
  - Background Accuracy          = TN / (TN + FP)
""")


if __name__ == '__main__':
    main()
