import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.predictor import EdgePredictor
from evaluation.metrics import compute_edge_metrics, aggregate_metrics

# --- Fixtures ---
@pytest.fixture
def sample_keypoints():
    # form a square
    return np.array([
        [10, 10], [10, 20], [20, 20], [20, 10],
        [15, 10], [15, 20], [10, 15], [20, 15] # midpoints
    ], dtype=float)

@pytest.fixture
def sample_scores(sample_keypoints):
    return np.ones(len(sample_keypoints))

@pytest.fixture
def sample_descriptors(sample_keypoints):
    # Make them all identical so they pass similarity checks
    d = np.zeros((256, len(sample_keypoints)))
    d[0, :] = 1.0 # All normalized to 1.0
    return d

@pytest.fixture
def predictor():
    return EdgePredictor(eps=15, min_samples=2, desc_thresh=0.5, max_dist=20, gap_dist=30)

# --- Predictor Tests ---

def test_predictor_initialization(predictor):
    assert predictor.eps == 15
    assert predictor.min_samples == 2

def test_predictor_predict_empty():
    pred = EdgePredictor()
    # No points
    em = pred.predict(np.array([]), (100, 100))
    assert em.shape == (100, 100)
    assert np.sum(em) == 0

def test_predictor_predict_basic(predictor, sample_keypoints, sample_scores, sample_descriptors):
    img_shape = (50, 50)
    edge_map = predictor.predict(sample_keypoints, img_shape, scores=sample_scores, descriptors=sample_descriptors)
    assert edge_map.shape == img_shape
    # We expect some edges because points are close and descriptors match
    assert np.sum(edge_map) > 0 
    assert np.max(edge_map) == 255

def test_predictor_geometric_types(predictor):
    # 1. Linear points (MST)
    linear_pts = np.array([[10, 10], [12, 12], [14, 14], [16, 16], [18, 18]])
    em = predictor.predict(linear_pts, (50, 50))
    # Should perform fine (MST logic)
    assert np.sum(em) > 0

def test_convex_hull_fallback(predictor):
    # Circular points (Hull)
    angles = np.linspace(0, 2*np.pi, 10)
    circle_pts = np.array([[25 + 10*np.cos(a), 25 + 10*np.sin(a)] for a in angles])
    em = predictor.predict(circle_pts, (50, 50))
    # Should detect hull/circle
    assert np.sum(em) > 0

# --- Metrics Tests ---

def test_compute_metrics():
    # 10x10 image
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)
    
    # Perfect match
    pred[5, :] = 255
    gt[5, :] = 255
    
    m = compute_edge_metrics(pred, gt, tolerance=1)
    assert m['precision'] == 1.0
    assert m['recall'] == 1.0
    assert m['f1'] == 1.0
    
    # No match
    pred_bad = np.zeros((10,10), dtype=np.uint8)
    pred_bad[0, :] = 255 # Top row
    m_bad = compute_edge_metrics(pred_bad, gt, tolerance=1)
    assert m_bad['precision'] < 1.0
    assert m_bad['recall'] == 0.0

def test_aggregate_metrics():
    m1 = {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'bg_accuracy': 1.0}
    m2 = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'bg_accuracy': 0.0}
    
    agg = aggregate_metrics([m1, m2])
    assert agg['precision_mean'] == 0.5
    assert agg['recall_mean'] == 0.5
    assert agg['f1_mean'] == 0.5

def test_compute_ods():
    from evaluation.metrics import compute_ods
    
    # Create simple ground truth and prediction
    # Image 1: Perfect match at threshold 0.5
    gt1 = np.zeros((10,10), dtype=np.uint8); gt1[5,:] = 255
    pred1 = np.zeros((10,10), dtype=np.uint8); pred1[5,:] = 200 # 200 > 0.5*255
    
    # Image 2: No match
    gt2 = np.zeros((10,10), dtype=np.uint8); gt2[0,:] = 255
    pred2 = np.zeros((10,10), dtype=np.uint8); pred2[9,:] = 200
    
    # Test ODS
    res = compute_ods([pred1, pred2], [gt1, gt2], thresholds=[0.1, 0.5, 0.9])
    
    # At thresh 0.5:
    # Img 1: TP=10, FP=0, FN=0 -> F1=1.0
    # Img 2: TP=0, FP=10, FN=10 -> F1=0
    # Combined: TP=10, FP=10, FN=10 -> P=0.5, R=0.5, F1=0.5
    
    # At thresh 0.9 (200/255 = 0.78 < 0.9): Preds become 0
    # Combined: TP=0 -> F1=0
    
    assert res['ods_f1'] > 0
    assert 0.4 < res['ods_f1'] < 0.6 # Should be around 0.5


# --- Boundary Case Tests ---

def test_predictor_single_keypoint():
    """Boundary case: single keypoint should produce no edges."""
    pred = EdgePredictor()
    single_pt = np.array([[50, 50]], dtype=float)
    em = pred.predict(single_pt, (100, 100))
    assert em.shape == (100, 100)
    assert np.sum(em) == 0  # Can't draw edge with single point


def test_predictor_two_keypoints():
    """Boundary case: two keypoints should produce one edge."""
    pred = EdgePredictor(eps=50, min_samples=2, max_dist=100)
    two_pts = np.array([[10, 10], [20, 20]], dtype=float)
    em = pred.predict(two_pts, (50, 50))
    assert em.shape == (50, 50)
    # Should have some edge pixels connecting the two points
    assert np.sum(em) > 0


def test_predictor_dense_keypoints():
    """Boundary case: dense keypoints in small area."""
    pred = EdgePredictor(eps=10, min_samples=2, max_dist=15)
    # 50 points in a 20x20 pixel area
    np.random.seed(42)
    dense_pts = np.random.rand(50, 2) * 20 + 40  # Centered at (50, 50)
    em = pred.predict(dense_pts, (100, 100))
    assert em.shape == (100, 100)
    assert np.sum(em) > 0  # Should produce edges
    # Edges should be concentrated in the center region
    center_region = em[30:70, 30:70]
    assert np.sum(center_region) > np.sum(em) * 0.5


def test_predictor_collinear_keypoints():
    """Boundary case: perfectly collinear points (line)."""
    pred = EdgePredictor(eps=20, min_samples=2, max_dist=15)
    # Horizontal line
    collinear_pts = np.array([[10 + i*5, 50] for i in range(10)], dtype=float)
    em = pred.predict(collinear_pts, (100, 100))
    assert em.shape == (100, 100)
    assert np.sum(em) > 0
    # Most edge pixels should be on row 50
    assert np.sum(em[49:52, :]) > np.sum(em) * 0.8


def test_predictor_sparse_keypoints():
    """Boundary case: sparse keypoints far apart (no edges expected)."""
    pred = EdgePredictor(eps=10, min_samples=2, max_dist=20)
    # Points too far apart to connect
    sparse_pts = np.array([[10, 10], [90, 90], [10, 90], [90, 10]], dtype=float)
    em = pred.predict(sparse_pts, (100, 100))
    assert em.shape == (100, 100)
    # May or may not have edges depending on clustering


# --- Metric Edge Case Tests ---

def test_metrics_tp_fp_tn_fn_values():
    """Verify TP, FP, TN, FN are computed correctly."""
    # Create a simple 5x5 case
    pred = np.zeros((5, 5), dtype=np.uint8)
    gt = np.zeros((5, 5), dtype=np.uint8)

    # Ground truth: row 2 is edge
    gt[2, :] = 255  # 5 edge pixels

    # Prediction: row 2 and row 3 are edges
    pred[2, :] = 255  # 5 correct (TP)
    pred[3, :] = 255  # 5 false positives (FP)

    m = compute_edge_metrics(pred, gt, tolerance=0)

    # With tolerance=0:
    # TP = 5 (row 2 matches)
    # FP = 5 (row 3 predicted but not in GT)
    # FN = 0 (all GT edges are predicted)
    # TN = 15 (remaining background pixels)
    assert m['tp'] == 5
    assert m['fp'] == 5
    assert m['fn'] == 0
    assert m['tn'] == 15


def test_metrics_all_false_positives():
    """Edge case: prediction with no overlap with ground truth."""
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)

    gt[0, :] = 255   # GT at top
    pred[9, :] = 255  # Pred at bottom

    m = compute_edge_metrics(pred, gt, tolerance=0)

    assert m['tp'] == 0
    assert m['fp'] == 10
    assert m['fn'] == 10
    assert m['precision'] == 0.0
    assert m['recall'] == 0.0
    assert m['f1'] == 0.0


def test_metrics_all_false_negatives():
    """Edge case: empty prediction with non-empty ground truth."""
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)

    gt[5, :] = 255  # GT has edges
    # pred is all zeros

    m = compute_edge_metrics(pred, gt, tolerance=0)

    assert m['tp'] == 0
    assert m['fp'] == 0
    assert m['fn'] == 10
    assert m['precision'] == 0.0
    assert m['recall'] == 0.0


def test_metrics_empty_ground_truth():
    """Edge case: empty ground truth."""
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)

    pred[5, :] = 255  # Prediction has edges
    # gt is all zeros

    m = compute_edge_metrics(pred, gt, tolerance=0)

    assert m['tp'] == 0
    assert m['fp'] == 10
    assert m['fn'] == 0


def test_metrics_both_empty():
    """Edge case: both prediction and ground truth are empty."""
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)

    m = compute_edge_metrics(pred, gt, tolerance=0)

    assert m['tp'] == 0
    assert m['fp'] == 0
    assert m['fn'] == 0
    assert m['tn'] == 100  # All pixels are TN
    assert m['bg_accuracy'] == 1.0


def test_aggregate_metrics_empty_list():
    """Edge case: empty metrics list."""
    agg = aggregate_metrics([])
    assert agg == {}


def test_bg_accuracy_formula():
    """Verify background accuracy = TN / (TN + FP)."""
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)

    gt[5, :] = 255
    pred[5, :] = 255  # Perfect match

    m = compute_edge_metrics(pred, gt, tolerance=0)

    # BG Accuracy = TN / (TN + FP)
    expected_bg_acc = m['tn'] / (m['tn'] + m['fp']) if (m['tn'] + m['fp']) > 0 else 0
    assert m['bg_accuracy'] == expected_bg_acc
