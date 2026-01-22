#!/usr/bin/env python3
"""
Main pipeline script for edge detection.

Usage:
    python scripts/run_pipeline.py                    # Run evaluation on all images
    python scripts/run_pipeline.py --method union     # Use specific method
    python scripts/run_pipeline.py --visualize        # Generate visualizations
    python scripts/run_pipeline.py --test             # Run tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import cv2

from model.predictor import EdgePredictor
from model.utils import (
    load_keypoints, filter_keypoints, load_image,
    load_edge_gt, get_image_paths, create_comparison_image
)
from evaluation.metrics import compute_edge_metrics, aggregate_metrics
from evaluation.evaluate import evaluate_dataset, compare_methods


def run_evaluation(data_dir, method='union', k=2, beta=0.5):
    """Run evaluation on the dataset."""
    print(f"\nRunning evaluation with method='{method}', k={k}, beta={beta}")
    print("=" * 60)

    results = evaluate_dataset(data_dir, method=method, k=k, beta=beta, verbose=True)
    return results


def run_comparison(data_dir):
    """Compare all methods."""
    print("\nComparing all methods...")
    results = compare_methods(data_dir, verbose=True)
    return results


def generate_visualizations(data_dir, output_dir, method='union', n_images=5):
    """Generate visualization images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    predictor = EdgePredictor(k=2, beta=0.5)
    paths = get_image_paths(data_dir)[:n_images]

    print(f"\nGenerating {len(paths)} visualizations with method='{method}'")
    print("=" * 60)

    for rgb_path, edge_path, kp_path in paths:
        img = load_image(rgb_path)
        h, w = img.shape[:2]

        gt_edge = load_edge_gt(edge_path, target_size=(w, h))

        keypoints, scores, _ = load_keypoints(kp_path)
        filtered_kp = filter_keypoints(keypoints, scores)

        pred_edge = predictor.predict(filtered_kp, (h, w), method=method)

        comparison = create_comparison_image(img, pred_edge, gt_edge)

        out_path = output_dir / f"{rgb_path.stem}_comparison.jpg"
        cv2.imwrite(str(out_path), comparison)
        print(f"  Saved: {out_path.name}")

    print(f"\nVisualizations saved to: {output_dir}")


def run_tests():
    """Run unit tests."""
    import unittest

    test_dir = Path(__file__).parent.parent / 'tests'
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description='Edge Detection Pipeline')
    parser.add_argument('--method', type=str, default='union',
                       choices=['union', 'intersection', 'knn_alpha', 'gauging_beta'],
                       help='Edge detection method (default: union)')
    parser.add_argument('--k', type=int, default=2,
                       help='Number of clusters for K-means (default: 2)')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Border detection threshold (default: 0.5)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all methods')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization images')
    parser.add_argument('--n-vis', type=int, default=5,
                       help='Number of visualizations to generate (default: 5)')
    parser.add_argument('--test', action='store_true',
                       help='Run unit tests')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Path to output directory for visualizations')

    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / 'data'
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'output'

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    if args.compare:
        run_comparison(data_dir)
    elif args.visualize:
        generate_visualizations(data_dir, output_dir, method=args.method, n_images=args.n_vis)
    else:
        run_evaluation(data_dir, method=args.method, k=args.k, beta=args.beta)


if __name__ == '__main__':
    main()
