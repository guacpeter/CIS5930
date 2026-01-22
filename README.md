# Edge Detection from SuperPoint Keypoints

Edge detection using hybrid approach combining K-means Alpha and Gauging-Beta algorithms.

## Project Structure

```
project/
├── data/
│   ├── rgb/           # Input RGB images (BIPED dataset)
│   └── edges/         # Ground truth edge images
├── superpoint/
│   └── extract_keypoints.py    # SuperPoint keypoint extraction
├── model/
│   ├── predictor.py   # Edge prediction (hybrid approach)
│   ├── geometry.py    # Geometric functions (alpha shape, Delaunay)
│   └── utils.py       # Utility functions
├── evaluation/
│   ├── metrics.py     # Precision, Recall, F1 metrics
│   └── evaluate.py    # Dataset evaluation
├── tests/
│   ├── test_metrics.py
│   └── test_predictor.py
├── scripts/
│   └── run_pipeline.py    # Main pipeline script
├── pyproject.toml
└── README.md
```

## Installation

```bash
pip install numpy opencv-python scipy scikit-learn torch
```

## Usage

### 1. Extract Keypoints (if needed)

```bash
python superpoint/extract_keypoints.py -i data/rgb -o data/keypoints
```

### 2. Run Evaluation

```bash
# Run with default settings (union method)
python scripts/run_pipeline.py

# Compare all methods
python scripts/run_pipeline.py --compare

# Generate visualizations
python scripts/run_pipeline.py --visualize

# Run tests
python scripts/run_pipeline.py --test
```

### 3. Use Programmatically

```python
from model.predictor import EdgePredictor
from model.utils import load_keypoints, filter_keypoints

# Load keypoints
keypoints, scores, _ = load_keypoints('data/keypoints/RGB_001_keypoints.npz')
filtered_kp = filter_keypoints(keypoints, scores)

# Predict edges
predictor = EdgePredictor(k=2, beta=0.5)
edge_map = predictor.predict(filtered_kp, (480, 640), method='union')
```

## Methods

| Method | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| kmeans_alpha | 0.607 | 0.700 | 0.642 |
| gauging_beta | 0.693 | 0.546 | 0.601 |
| **union** | 0.609 | 0.708 | **0.647** |
| intersection | 0.693 | 0.533 | 0.592 |

Best method: **union** (combining K-means Alpha + Gauging-Beta edges)

## References

- SuperPoint: Self-Supervised Interest Point Detection and Description
- BIPED: Barcelona Images for Perceptual Edge Detection
- Gauging-β: Border-aware hierarchical clustering based on density and proximity (Pattern Recognition, 2025)
