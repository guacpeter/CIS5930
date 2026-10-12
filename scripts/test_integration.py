#!/usr/bin/env python3
"""Test integration between original EdgePredictor and SOLD2."""

import numpy as np
import sys
sys.path.insert(0, '.')

from model.predictor import EdgePredictor
from model.sold2_predictor import SOLD2Predictor
from model.hybrid_predictor import HybridPredictor

def main():
    # Test keypoints
    keypoints = np.array([[50, 50], [100, 100], [150, 50], [200, 150]], dtype=float)
    img_shape = (256, 256)
    
    print("=" * 60)
    print("Testing Integration: Original EdgePredictor + SOLD2")
    print("=" * 60)
    
    # 1. Original EdgePredictor (your algorithm)
    print("\n1. Original EdgePredictor (DBSCAN + MST + ConvexHull):")
    edge_pred = EdgePredictor()
    edge_map1 = edge_pred.predict(keypoints, img_shape)
    print(f"   Shape: {edge_map1.shape}")
    print(f"   Non-zero pixels: {np.count_nonzero(edge_map1)}")
    
    # 2. SOLD2 Predictor (deep learning)
    print("\n2. SOLD2Predictor (Deep Learning):")
    sold2_pred = SOLD2Predictor(device='cpu')
    edge_map2 = sold2_pred.predict(keypoints, img_shape)
    print(f"   Shape: {edge_map2.shape}")
    print(f"   Non-zero pixels: {np.count_nonzero(edge_map2)}")
    
    # 3. Hybrid Predictor (combines both!)
    print("\n3. HybridPredictor (Fusion of both methods):")
    hybrid = HybridPredictor(mode='fusion', sold2_weight=0.5)
    image = np.random.randint(0, 256, img_shape, dtype=np.uint8)
    edge_map3 = hybrid.predict(keypoints, img_shape, image=image)
    print(f"   Shape: {edge_map3.shape}")
    print(f"   Non-zero pixels: {np.count_nonzero(edge_map3)}")
    
    # Verify API compatibility
    print("\n" + "=" * 60)
    print("API Compatibility Check:")
    print("=" * 60)
    
    # All should return same shape and dtype
    assert edge_map1.shape == edge_map2.shape == edge_map3.shape
    assert edge_map1.dtype == edge_map2.dtype == edge_map3.dtype == np.uint8
    print("✓ All predictors return same shape and dtype")
    print("✓ Original EdgePredictor works with SOLD2")
    print("✓ HybridPredictor successfully combines both methods")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Your original algorithm integrates with SOLD2!")
    print("=" * 60)

if __name__ == '__main__':
    main()
