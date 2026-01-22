"""
Metrics for edge detection evaluation.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_edge_metrics(pred_edge, gt_edge, tolerance=5):
    """
    Compute precision, recall, and F1 score for edge detection.

    Uses distance transform to allow tolerance in matching.

    Args:
        pred_edge: Predicted edge map (binary or grayscale)
        gt_edge: Ground truth edge map (binary or grayscale)
        tolerance: Pixel tolerance for matching (default: 5)

    Returns:
        dict with 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'
    """
    pred = (pred_edge > 127).astype(np.uint8)
    gt = (gt_edge > 127).astype(np.uint8)

    gt_dist = distance_transform_edt(1 - gt)
    pred_dist = distance_transform_edt(1 - pred)

    pred_points = np.where(pred == 1)
    tp = np.sum(gt_dist[pred_points] <= tolerance)
    fp = np.sum(gt_dist[pred_points] > tolerance)

    gt_points = np.where(gt == 1)
    fn = np.sum(pred_dist[gt_points] > tolerance)

    # Calculate True Negatives (TN)
    # TN = Total Pixels - (TP + FP + FN)
    # Note: This is an approximation for edge detection where "negatives" are the vast majority
    total_pixels = pred.shape[0] * pred.shape[1]
    tn = total_pixels - (tp + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Background Accuracy = TN / (TN + FP)
    # Measures how many background pixels were correctly identified as background
    bg_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'bg_accuracy': bg_accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def compute_ods(predictions, ground_truths, thresholds=None, tolerance=5):
    """
    Compute Optimal Dataset Scale (ODS) F-measure.

    Finds the best threshold across the entire dataset.

    Args:
        predictions: List of predicted edge maps
        ground_truths: List of ground truth edge maps
        thresholds: List of thresholds to try (default: 0.1 to 0.9)
        tolerance: Pixel tolerance for matching

    Returns:
        dict with 'ods_f1', 'best_threshold', 'precision', 'recall'
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    best_f1 = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0

    for thresh in thresholds:
        all_tp, all_fp, all_fn = 0, 0, 0

        for pred, gt in zip(predictions, ground_truths):
            pred_binary = (pred > thresh * 255).astype(np.uint8)
            metrics = compute_edge_metrics(pred_binary * 255, gt, tolerance)
            all_tp += metrics['tp']
            all_fp += metrics['fp']
            all_fn += metrics['fn']

        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_precision = precision
            best_recall = recall

    return {
        'ods_f1': best_f1,
        'best_threshold': best_threshold,
        'precision': best_precision,
        'recall': best_recall
    }


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics from multiple images.

    Args:
        metrics_list: List of metric dicts from compute_edge_metrics

    Returns:
        dict with averaged metrics and standard deviations
    """
    if not metrics_list:
        return {}

    precisions = [m['precision'] for m in metrics_list]
    recalls = [m['recall'] for m in metrics_list]
    f1s = [m['f1'] for m in metrics_list]
    bg_accs = [m['bg_accuracy'] for m in metrics_list]

    return {
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s),
        'bg_accuracy_mean': np.mean(bg_accs),
        'bg_accuracy_std': np.std(bg_accs),
        'n_images': len(metrics_list)
    }
