"""
Hybrid Edge Predictor: SOLD2 + SuperPoint Integration.

This predictor combines two complementary approaches:
1. SOLD2: Deep learning-based dense line detection (high line coverage)
2. SuperPoint/EdgePredictor: Keypoint-based geometric structure (precise keypoint connections)

The fusion strategy:
- Use SOLD2 for dense edge/line heatmap
- Use SuperPoint keypoints to guide and refine line connections
- Combine both outputs with weighted fusion or intersection

This addresses limitations of each approach:
- SOLD2 alone: May over-predict edges, lacks keypoint precision
- SuperPoint alone: May miss subtle edges, depends on keypoint density
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict

from .predictor import EdgePredictor
from .sold2_network import SOLD2Net


class HybridPredictor:
    """
    Hybrid edge predictor combining SOLD2 and SuperPoint-based EdgePredictor.
    
    Modes:
    - 'fusion': Weighted combination of SOLD2 and EdgePredictor outputs
    - 'intersection': Only edges detected by both methods
    - 'sold2_guided': Use SOLD2 heatmap to validate EdgePredictor edges
    - 'keypoint_guided': Use SuperPoint keypoints to refine SOLD2 lines
    """
    
    def __init__(self,
                 mode: str = 'fusion',
                 sold2_weight: float = 0.5,
                 sold2_checkpoint: Optional[str] = None,
                 sold2_thresh: float = 0.3,
                 device: str = 'cpu',
                 # EdgePredictor params
                 eps: float = 20,
                 min_samples: int = 3,
                 desc_thresh: float = 0.5,
                 max_dist: float = 20,
                 gap_dist: float = 60,
                 # SOLD2 model params (for lightweight)
                 base_channels: int = 16,
                 descriptor_dim: int = 32):
        """
        Initialize hybrid predictor.
        
        Args:
            mode: Fusion mode ('fusion', 'intersection', 'sold2_guided', 'keypoint_guided')
            sold2_weight: Weight for SOLD2 output in fusion mode (0-1)
            sold2_checkpoint: Path to trained SOLD2 checkpoint
            sold2_thresh: Threshold for SOLD2 line heatmap
            device: 'cuda' or 'cpu'
            eps, min_samples, etc.: EdgePredictor parameters
            base_channels, descriptor_dim: SOLD2 model architecture params
        """
        self.mode = mode
        self.sold2_weight = sold2_weight
        self.sold2_thresh = sold2_thresh
        self.device = device
        
        # Initialize EdgePredictor (geometric + SuperPoint-based)
        self.edge_predictor = EdgePredictor(
            eps=eps,
            min_samples=min_samples,
            desc_thresh=desc_thresh,
            max_dist=max_dist,
            gap_dist=gap_dist
        )
        
        # Initialize SOLD2 model
        self.sold2_model = SOLD2Net(
            in_channels=1,
            base_channels=base_channels,
            descriptor_dim=descriptor_dim
        )
        
        # Load checkpoint if provided
        if sold2_checkpoint and Path(sold2_checkpoint).exists():
            checkpoint = torch.load(sold2_checkpoint, map_location=device)
            self.sold2_model.load_state_dict(checkpoint)
            print(f"Loaded SOLD2 checkpoint from {sold2_checkpoint}")
        
        self.sold2_model = self.sold2_model.to(device)
        self.sold2_model.eval()
    
    def predict(self,
                keypoints: np.ndarray,
                img_shape: Tuple[int, int],
                scores: Optional[np.ndarray] = None,
                descriptors: Optional[np.ndarray] = None,
                image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict edge map using hybrid approach.
        
        Args:
            keypoints: (N, 2) SuperPoint keypoint coordinates (x, y)
            img_shape: (H, W) output shape
            scores: Optional keypoint confidence scores
            descriptors: (256, N) SuperPoint descriptors
            image: Optional input image for SOLD2
            
        Returns:
            Binary edge map (H, W) with values 0 or 255
        """
        H, W = img_shape[:2]
        
        # Get EdgePredictor output (SuperPoint-based)
        edge_map_superpoint = self.edge_predictor.predict(
            keypoints, img_shape, scores, descriptors
        )
        
        # Get SOLD2 output (if image provided)
        if image is not None:
            edge_map_sold2 = self._get_sold2_edges(image)
            # Resize to match
            if edge_map_sold2.shape != (H, W):
                edge_map_sold2 = cv2.resize(edge_map_sold2, (W, H), 
                                            interpolation=cv2.INTER_NEAREST)
        else:
            # Without image, use only SuperPoint
            return edge_map_superpoint
        
        # Combine based on mode
        if self.mode == 'fusion':
            return self._fuse_weighted(edge_map_sold2, edge_map_superpoint)
        elif self.mode == 'intersection':
            return self._fuse_intersection(edge_map_sold2, edge_map_superpoint)
        elif self.mode == 'sold2_guided':
            return self._sold2_guided(edge_map_sold2, edge_map_superpoint, keypoints)
        elif self.mode == 'keypoint_guided':
            return self._keypoint_guided(edge_map_sold2, keypoints, image)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _get_sold2_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Get edge map from SOLD2 model.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            
        Returns:
            Binary edge map (H, W)
        """
        # Preprocess
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        H, W = gray.shape
        
        # Resize for network
        new_H = ((H + 31) // 32) * 32
        new_W = ((W + 31) // 32) * 32
        gray_resized = cv2.resize(gray, (new_W, new_H))
        
        # Normalize and convert to tensor
        img_tensor = torch.from_numpy(gray_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.sold2_model(img_tensor)
        
        # Get line heatmap and threshold
        line_heatmap = outputs['line_heatmap'][0, 0].cpu().numpy()
        edge_map = (line_heatmap > self.sold2_thresh).astype(np.uint8) * 255
        
        # Resize back
        edge_map = cv2.resize(edge_map, (W, H), interpolation=cv2.INTER_NEAREST)
        
        return edge_map
    
    def _fuse_weighted(self, 
                       sold2_map: np.ndarray, 
                       superpoint_map: np.ndarray) -> np.ndarray:
        """
        Weighted fusion of SOLD2 and SuperPoint edge maps.
        
        The fusion uses weighted averaging, then thresholding.
        """
        # Convert to float
        sold2_float = sold2_map.astype(float) / 255.0
        superpoint_float = superpoint_map.astype(float) / 255.0
        
        # Weighted combination
        combined = (self.sold2_weight * sold2_float + 
                   (1 - self.sold2_weight) * superpoint_float)
        
        # Threshold at 0.3 (any contribution is enough)
        result = (combined > 0.3).astype(np.uint8) * 255
        
        return result
    
    def _fuse_intersection(self,
                           sold2_map: np.ndarray,
                           superpoint_map: np.ndarray) -> np.ndarray:
        """
        Intersection fusion: only edges detected by both methods.
        
        This is conservative and produces high-precision results.
        """
        # Dilate both maps slightly to allow near-matches
        kernel = np.ones((3, 3), np.uint8)
        sold2_dilated = cv2.dilate(sold2_map, kernel, iterations=1)
        superpoint_dilated = cv2.dilate(superpoint_map, kernel, iterations=1)
        
        # Intersection
        intersection = cv2.bitwise_and(sold2_dilated, superpoint_dilated)
        
        return intersection
    
    def _sold2_guided(self,
                      sold2_map: np.ndarray,
                      superpoint_map: np.ndarray,
                      keypoints: np.ndarray) -> np.ndarray:
        """
        Use SOLD2 heatmap to validate SuperPoint-based edges.
        
        Only keep SuperPoint edges that are supported by SOLD2 detections.
        """
        # Dilate SOLD2 map to create a wider "support region"
        kernel = np.ones((5, 5), np.uint8)
        sold2_support = cv2.dilate(sold2_map, kernel, iterations=2)
        
        # Keep SuperPoint edges only where SOLD2 has support
        result = cv2.bitwise_and(superpoint_map, sold2_support)
        
        return result
    
    def _keypoint_guided(self,
                         sold2_map: np.ndarray,
                         keypoints: np.ndarray,
                         image: np.ndarray) -> np.ndarray:
        """
        Use SuperPoint keypoints to refine SOLD2 line detection.
        
        This snaps SOLD2 lines to keypoint locations for precision.
        """
        H, W = sold2_map.shape
        result = np.zeros((H, W), dtype=np.uint8)
        
        # Get SOLD2 line contours
        contours, _ = cv2.findContours(sold2_map, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        # For each keypoint, find nearby SOLD2 lines and draw connections
        for i in range(len(keypoints)):
            x1, y1 = keypoints[i].astype(int)
            if not (0 <= x1 < W and 0 <= y1 < H):
                continue
                
            # Check if keypoint is near a SOLD2 edge
            # Dilate point region and check intersection
            kp_region = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(kp_region, (x1, y1), 10, 255, -1)
            
            overlap = cv2.bitwise_and(sold2_map, kp_region)
            if np.sum(overlap) > 0:
                # Keypoint is near SOLD2 edge, connect to nearby keypoints
                for j in range(i + 1, len(keypoints)):
                    x2, y2 = keypoints[j].astype(int)
                    if not (0 <= x2 < W and 0 <= y2 < H):
                        continue
                    
                    # Check if path is supported by SOLD2
                    path_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.line(path_mask, (x1, y1), (x2, y2), 255, 3)
                    
                    # Check overlap with SOLD2
                    path_overlap = cv2.bitwise_and(sold2_map, path_mask)
                    coverage = np.sum(path_overlap) / (np.sum(path_mask) + 1e-5)
                    
                    if coverage > 0.3:  # At least 30% coverage
                        cv2.line(result, (x1, y1), (x2, y2), 255, 1)
        
        return result
    
    def get_sold2_heatmap(self, image: np.ndarray) -> np.ndarray:
        """
        Get raw SOLD2 line heatmap (for visualization).
        
        Returns:
            Float heatmap (H, W) with values in [0, 1]
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        H, W = gray.shape
        new_H = ((H + 31) // 32) * 32
        new_W = ((W + 31) // 32) * 32
        gray_resized = cv2.resize(gray, (new_W, new_H))
        
        img_tensor = torch.from_numpy(gray_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.sold2_model(img_tensor)
        
        heatmap = outputs['line_heatmap'][0, 0].cpu().numpy()
        heatmap = cv2.resize(heatmap, (W, H))
        
        return heatmap


def create_hybrid_predictor(checkpoint_path: Optional[str] = None,
                            mode: str = 'fusion',
                            **kwargs) -> HybridPredictor:
    """
    Factory function to create HybridPredictor with sensible defaults.
    
    Args:
        checkpoint_path: Path to SOLD2 checkpoint (defaults to trained lightweight model)
        mode: Fusion mode
        **kwargs: Additional parameters for HybridPredictor
        
    Returns:
        Configured HybridPredictor instance
    """
    if checkpoint_path is None:
        # Try default trained model path
        default_path = Path(__file__).parent.parent / 'results' / 'sold2' / 'sold2_synth_best.pt'
        if default_path.exists():
            checkpoint_path = str(default_path)
    
    return HybridPredictor(
        mode=mode,
        sold2_checkpoint=checkpoint_path,
        **kwargs
    )
