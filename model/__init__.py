"""
Edge detection model package.

Includes:
- EdgePredictor: Geometric edge detection (DBSCAN + MST + Convex Hull)
- SOLD2Predictor: Unsupervised deep learning line detection
- SOLD2Net: SOLD2 neural network architecture
- KeypointConditionedSOLD2: SOLD2 with SuperPoint keypoint input
- KeypointSOLD2Predictor: Predictor using keypoint-conditioned SOLD2
"""

from .predictor import EdgePredictor
from .sold2_predictor import SOLD2Predictor
from .sold2_network import SOLD2Net, KeypointConditionedSOLD2
from .keypoint_sold2_predictor import KeypointSOLD2Predictor

__all__ = [
    'EdgePredictor',
    'SOLD2Predictor',
    'SOLD2Net',
    'KeypointConditionedSOLD2',
    'KeypointSOLD2Predictor',
]
