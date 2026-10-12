"""
SOLD2 Predictor with SuperPoint Keypoint Conditioning.

This predictor uses SuperPoint keypoints as additional input features
to guide SOLD2's line detection, creating a hybrid approach that
leverages both keypoint detection and dense line prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from .sold2_network import KeypointConditionedSOLD2, create_keypoint_sold2_model


class KeypointSOLD2Predictor:
    """
    SOLD2 line detector conditioned on SuperPoint keypoints.
    
    This predictor takes both an image and SuperPoint keypoints,
    using the keypoints to guide line detection. Lines are more
    likely to be detected connecting keypoint locations.
    
    Usage:
        predictor = KeypointSOLD2Predictor()
        
        # Load keypoints from SuperPoint
        keypoints = np.load('keypoints.npz')['keypoints']
        scores = np.load('keypoints.npz')['scores']
        
        # Detect edges using both image and keypoints
        edge_map = predictor.predict(keypoints, img_shape, scores=scores, image=image)
    """
    
    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None,
                 junction_thresh: float = 0.5,
                 line_thresh: float = 0.3,
                 nms_radius: int = 4,
                 min_line_length: int = 10,
                 max_num_lines: int = 200,
                 base_channels: int = 64,
                 descriptor_dim: int = 128,
                 keypoint_sigma: float = 4.0):
        """
        Initialize keypoint-conditioned SOLD2 predictor.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: 'cuda', 'cpu', or None for auto-detection
            junction_thresh: Threshold for junction detection
            line_thresh: Threshold for line heatmap
            nms_radius: Radius for non-maximum suppression
            min_line_length: Minimum line segment length in pixels
            max_num_lines: Maximum number of lines to return
            base_channels: Base channels for the network
            descriptor_dim: Descriptor dimension
            keypoint_sigma: Sigma for Gaussian keypoint blobs
        """
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Store dimensions
        self.descriptor_dim = descriptor_dim
        self.keypoint_sigma = keypoint_sigma
        
        # Model setup
        self.model = KeypointConditionedSOLD2(
            base_channels=base_channels,
            descriptor_dim=descriptor_dim,
            keypoint_sigma=keypoint_sigma
        )
        
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Detection parameters
        self.junction_thresh = junction_thresh
        self.line_thresh = line_thresh
        self.nms_radius = nms_radius
        self.min_line_length = min_line_length
        self.max_num_lines = max_num_lines
    
    def predict(self, 
                keypoints: np.ndarray,
                img_shape: Tuple[int, int],
                scores: Optional[np.ndarray] = None,
                descriptors: Optional[np.ndarray] = None,
                image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict edge map using keypoint-conditioned SOLD2.
        
        Args:
            keypoints: (N, 2) array of keypoint coordinates (x, y)
            img_shape: (H, W) output shape
            scores: (N,) keypoint confidence scores
            descriptors: Unused, for API compatibility
            image: Input image (H, W) or (H, W, 3) - REQUIRED for this predictor
            
        Returns:
            Binary edge map (H, W) with values 0 or 255
        """
        H, W = img_shape[:2]
        
        if image is None:
            # Without image, fall back to simple keypoint-based edges
            edge_map = np.zeros((H, W), dtype=np.uint8)
            if len(keypoints) < 2:
                return edge_map
            # Draw lines between nearby keypoints as fallback
            for i in range(len(keypoints)):
                for j in range(i + 1, min(i + 5, len(keypoints))):
                    x1, y1 = int(keypoints[i][0]), int(keypoints[i][1])
                    x2, y2 = int(keypoints[j][0]), int(keypoints[j][1])
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if dist < 50:  # Only connect nearby keypoints
                        cv2.line(edge_map, (x1, y1), (x2, y2), 255, 1)
            return edge_map
        
        # Default scores if not provided
        if scores is None:
            scores = np.ones(len(keypoints))
        
        # Preprocess image
        image_tensor = self._preprocess(image)
        _, _, H_net, W_net = image_tensor.shape
        
        # Create keypoint map
        keypoint_map = self.model.create_keypoint_map(
            keypoints, scores, (H_net, W_net), self.device
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor, keypoint_map=keypoint_map)
        
        # Get line heatmap
        line_heatmap = outputs['line_heatmap'][0, 0].cpu().numpy()
        
        # Threshold to binary
        edge_map = (line_heatmap > self.line_thresh).astype(np.uint8) * 255
        
        # Resize to original size if needed
        if edge_map.shape != (H, W):
            edge_map = cv2.resize(edge_map, (W, H), interpolation=cv2.INTER_NEAREST)
        
        return edge_map
    
    def detect_from_image(self, 
                          image: np.ndarray,
                          keypoints: np.ndarray,
                          scores: Optional[np.ndarray] = None) -> List[Tuple[int, int, int, int]]:
        """
        Detect line segments from image with keypoint conditioning.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            keypoints: (N, 2) keypoint coordinates (x, y)
            scores: (N,) keypoint scores
            
        Returns:
            List of (y1, x1, y2, x2) line segments
        """
        H_orig, W_orig = image.shape[:2]
        
        if scores is None:
            scores = np.ones(len(keypoints))
        
        # Preprocess
        image_tensor = self._preprocess(image)
        _, _, H_net, W_net = image_tensor.shape
        
        # Create keypoint map
        keypoint_map = self.model.create_keypoint_map(
            keypoints, scores, (H_net, W_net), self.device
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor, keypoint_map=keypoint_map)
        
        # Extract junctions
        junc_map = outputs['junctions'][0, 0]
        line_map = outputs['line_heatmap'][0, 0]
        
        junctions = self._extract_junctions(junc_map)
        
        # Match junctions to form lines
        lines = self._match_junctions(junctions, line_map)
        
        # Scale back to original size
        scale_y = H_orig / H_net
        scale_x = W_orig / W_net
        
        scaled_lines = []
        for y1, x1, y2, x2 in lines:
            scaled_lines.append((
                int(y1 * scale_y),
                int(x1 * scale_x),
                int(y2 * scale_y),
                int(x2 * scale_x)
            ))
        
        return scaled_lines[:self.max_num_lines]
    
    def get_keypoint_attention(self, 
                               image: np.ndarray,
                               keypoints: np.ndarray,
                               scores: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the keypoint attention/conditioning map for visualization.
        
        Args:
            image: Input image
            keypoints: SuperPoint keypoints
            scores: Keypoint scores
            
        Returns:
            Keypoint attention heatmap (H, W)
        """
        if scores is None:
            scores = np.ones(len(keypoints))
        
        image_tensor = self._preprocess(image)
        _, _, H_net, W_net = image_tensor.shape
        
        keypoint_map = self.model.create_keypoint_map(
            keypoints, scores, (H_net, W_net), self.device
        )
        
        return keypoint_map[0, 0].cpu().numpy()
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Normalize to [0, 1]
        if gray.max() > 1:
            gray = gray.astype(np.float32) / 255.0
        
        # Resize to multiple of 32
        H, W = gray.shape
        new_H = ((H + 31) // 32) * 32
        new_W = ((W + 31) // 32) * 32
        
        if (H, W) != (new_H, new_W):
            gray = cv2.resize(gray, (new_W, new_H))
        
        # Convert to tensor
        tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _extract_junctions(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Extract junction points from heatmap using NMS."""
        H, W = heatmap.shape
        pad = self.nms_radius
        
        heatmap_pad = F.pad(heatmap.unsqueeze(0).unsqueeze(0),
                           (pad, pad, pad, pad), mode='constant', value=0)
        max_pool = F.max_pool2d(heatmap_pad, kernel_size=2*pad+1, stride=1, padding=0)
        max_pool = max_pool.squeeze()
        
        is_peak = (heatmap == max_pool) & (heatmap > self.junction_thresh)
        
        return torch.nonzero(is_peak, as_tuple=False)
    
    def _match_junctions(self,
                         junctions: torch.Tensor,
                         line_heatmap: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        """Match junction pairs to form line segments."""
        lines = []
        n = len(junctions)
        
        if n < 2:
            return lines
        
        H, W = line_heatmap.shape
        
        for i in range(n):
            for j in range(i + 1, n):
                p1 = junctions[i].float()
                p2 = junctions[j].float()
                
                # Check line length
                length = torch.norm(p2 - p1).item()
                if length < self.min_line_length:
                    continue
                
                # Sample points along line
                num_samples = max(int(length), 10)
                t = torch.linspace(0, 1, num_samples, device=p1.device)
                
                ys = (p1[0] + t * (p2[0] - p1[0])).long().clamp(0, H - 1)
                xs = (p1[1] + t * (p2[1] - p1[1])).long().clamp(0, W - 1)
                
                # Check average line probability
                probs = line_heatmap[ys, xs]
                avg_prob = probs.mean().item()
                
                if avg_prob > self.line_thresh:
                    y1, x1 = junctions[i].tolist()
                    y2, x2 = junctions[j].tolist()
                    lines.append((y1, x1, y2, x2))
        
        return lines
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
