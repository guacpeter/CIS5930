"""
Utility functions for edge detection pipeline.

Provides functions for loading keypoints, images, ground truth edges,
and filtering/processing data.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional


def load_keypoints(kp_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load keypoints from an .npz file.

    Args:
        kp_path: Path to the .npz file containing keypoints.

    Returns:
        Tuple of (keypoints, scores, descriptors):
            - keypoints: (N, 2) array of (x, y) coordinates
            - scores: (N,) array of confidence scores
            - descriptors: (256, N) array of descriptors
    """
    data = np.load(kp_path)
    keypoints = data['keypoints']  # (N, 2)
    scores = data['scores']  # (N,)
    descriptors = data['descriptors']  # (256, N)
    return keypoints, scores, descriptors


def filter_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    score_thresh: float = 0.005,
    max_keypoints: Optional[int] = None
) -> np.ndarray:
    """
    Filter keypoints by score threshold and optionally limit count.

    Args:
        keypoints: (N, 2) array of keypoint coordinates.
        scores: (N,) array of confidence scores.
        score_thresh: Minimum score threshold (default: 0.005).
        max_keypoints: Maximum number of keypoints to return (optional).

    Returns:
        Filtered keypoints array (M, 2) where M <= N.
    """
    mask = scores >= score_thresh
    filtered_kp = keypoints[mask]
    filtered_scores = scores[mask]

    if max_keypoints is not None and len(filtered_kp) > max_keypoints:
        # Sort by score and take top k
        top_indices = np.argsort(filtered_scores)[::-1][:max_keypoints]
        filtered_kp = filtered_kp[top_indices]

    return filtered_kp


def load_image(img_path: str) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        img_path: Path to the image file.

    Returns:
        BGR image as numpy array.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    return img


def load_edge_gt(
    edge_path: str,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load ground truth edge map.

    Args:
        edge_path: Path to the edge image file.
        target_size: Optional (width, height) to resize to.

    Returns:
        Grayscale edge map as numpy array.
    """
    edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
    if edge is None:
        raise ValueError(f"Could not load edge image: {edge_path}")

    if target_size is not None:
        edge = cv2.resize(edge, target_size, interpolation=cv2.INTER_NEAREST)

    return edge


def get_image_paths(data_dir: str) -> List[Tuple[Path, Path, Path]]:
    """
    Get paths to RGB images, edge ground truth, and keypoint files.

    Args:
        data_dir: Path to the data directory containing rgb/, edges/, keypoints/.

    Returns:
        List of tuples (rgb_path, edge_path, keypoints_path).
    """
    data_dir = Path(data_dir)
    rgb_dir = data_dir / 'rgb'
    edge_dir = data_dir / 'edges'
    kp_dir = data_dir / 'keypoints'

    paths = []
    for rgb_path in sorted(rgb_dir.glob('*.jpg')):
        stem = rgb_path.stem  # e.g., "RGB_001"
        edge_path = edge_dir / f"{stem}.png"
        kp_path = kp_dir / f"{stem}_keypoints.npz"

        if edge_path.exists() and kp_path.exists():
            paths.append((rgb_path, edge_path, kp_path))

    return paths


def create_comparison_image(
    img: np.ndarray,
    pred_edge: np.ndarray,
    gt_edge: np.ndarray,
    keypoints: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create a side-by-side comparison image.

    Args:
        img: Original BGR image.
        pred_edge: Predicted edge map.
        gt_edge: Ground truth edge map.
        keypoints: Optional keypoints to overlay.

    Returns:
        Concatenated comparison image.
    """
    h, w = img.shape[:2]

    # Convert edges to BGR for visualization
    pred_bgr = cv2.cvtColor(pred_edge, cv2.COLOR_GRAY2BGR)
    gt_bgr = cv2.cvtColor(gt_edge, cv2.COLOR_GRAY2BGR)

    # Overlay keypoints on original image if provided
    img_with_kp = img.copy()
    if keypoints is not None:
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(img_with_kp, (x, y), 2, (0, 255, 0), -1)

    # Create comparison
    comparison = np.hstack([img_with_kp, pred_bgr, gt_bgr])

    return comparison
