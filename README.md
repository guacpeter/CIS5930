# Evaluation results and pictures arein results/ 

# Edge Detection from SuperPoint Keypoints

Edge detection using a hybrid geometric approach combining DBSCAN clustering, Minimum Spanning Tree (MST), and Convex Hull algorithms with semantic filtering.

## Project Structure

```
CIS5930/
├── data/
│   ├── rgb/                    # Input RGB images (BIPED dataset)
│   ├── edges/                  # Ground truth edge images
│   └── keypoints/              # Extracted SuperPoint keypoints (.npz)
├── model/
│   ├── __init__.py
│   ├── predictor.py            # EdgePredictor class (DBSCAN + MST + Convex Hull)
│   └── utils.py                # Utility functions (load_keypoints, filter_keypoints, etc.)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py              # Evaluation metrics (TP, FP, TN, FN, Precision, Recall, F1)
├── scripts/
│   ├── evaluation_metrics.py   # Main evaluation script with Excel export
│   ├── run_pipeline.py         # Pipeline runner
│   ├── test_pipeline.py        # Pipeline testing
│   └── test_and_visualize.py   # Visualization utilities
├── tests/
│   ├── __init__.py
│   └── test_model.py           # Unit tests (90% coverage)
├── superpoint/
│   ├── extract_keypoints.py    # SuperPoint keypoint extraction script
│   └── superpoint_repo/        # SuperPoint model implementation
├── results/
│   └── evaluation_metrics.xlsx # Exported evaluation results
├── pyproject.toml
├── Makefile
└── README.md
```

## Algorithm Overview

### Edge Prediction Pipeline

The `EdgePredictor` class uses a two-stage approach:

**Stage 1: Geometric Structural Proposals**
1. **DBSCAN Clustering** - Groups keypoints by spatial density
2. **PCA Analysis** - Determines cluster shape (linear vs. circular)
3. **Strategy Selection** based on cluster geometry:
   - **Linear clusters** (PCA ratio > 0.85) → **Minimum Spanning Tree (MST)**
   - **Circular/dense clusters** (PCA ratio < 0.3) → **Convex Hull**
   - **Generic blobs** → **K-Nearest Neighbors proximity**
4. **Gap Filling** - Connects MST endpoints within 30° alignment

**Stage 2: Semantic Filtering**
- Hull edges: Trusted without verification (strong geometric evidence)
- MST edges: Require descriptor similarity > 0.4
- Proximity edges: Require descriptor similarity > 0.75
- Gap-fill edges: Require descriptor similarity > 0.6

### Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **TP** | - | Correctly predicted edge pixels |
| **FP** | - | Pixels predicted as edges but are background |
| **TN** | - | Correctly predicted background pixels |
| **FN** | - | Missed edge pixels |
| **Precision** | TP / (TP + FP) | Edge correctness |
| **Recall** | TP / (TP + FN) | Edge coverage |
| **F1 Score** | 2 × P × R / (P + R) | Harmonic mean of precision and recall |
| **BG Accuracy** | TN / (TN + FP) | Background classification accuracy |

## Installation

```bash
# Install dependencies
pip install numpy opencv-python scipy scikit-learn pandas openpyxl

# For SuperPoint keypoint extraction (optional)
pip install torch torchvision
```

## Usage

### 1. Extract Keypoints (if needed)

```bash
python superpoint/extract_keypoints.py -i data/rgb -o data/keypoints
```

### 2. Run Evaluation

```bash
# Run full evaluation with metrics export to Excel
python scripts/evaluation_metrics.py

# Run pipeline with visualization
python scripts/run_pipeline.py --visualize
```

### 3. Run Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=model --cov=evaluation --cov-report=term-missing

# Expected output: 20 tests, 90% coverage
```

### 4. Use Programmatically

```python
from model.predictor import EdgePredictor
from model.utils import load_keypoints, filter_keypoints

# Load keypoints
keypoints, scores, descriptors = load_keypoints('data/keypoints/RGB_001_keypoints.npz')
filtered_kp = filter_keypoints(keypoints, scores)

# Initialize predictor
predictor = EdgePredictor(
    eps=20,           # DBSCAN epsilon (neighborhood radius)
    min_samples=3,    # DBSCAN minimum samples
    max_dist=20,      # Maximum edge distance
    gap_dist=60       # Gap filling distance
)

# Predict edges
edge_map = predictor.predict(filtered_kp, (480, 640), descriptors=descriptors)
```

## Output

The evaluation script generates:
- **Console output**: Per-image statistics, top 5 best/worst images by F1 score
- **Excel file** (`results/evaluation_metrics.xlsx`):
  - Per-method sheets with all image statistics
  - Top 5 best/worst images per method
  - Summary comparison across methods

## Test Coverage

| Module | Coverage |
|--------|----------|
| `evaluation/metrics.py` | 98% |
| `model/predictor.py` | 86% |
| **Total** | **90%** |

Tests cover:
- Metric computation (TP, FP, TN, FN verification)
- Boundary cases (empty, single, dense, sparse keypoints)
- Edge map generation logic
- Aggregate metrics calculation

## References

- [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)
- [BIPED: Barcelona Images for Perceptual Edge Detection](https://github.com/xavysp/BIPED)
- DBSCAN: Density-Based Spatial Clustering of Applications with Noise
- Minimum Spanning Tree (Kruskal's Algorithm)
- Quickhull Algorithm for Convex Hull computation
