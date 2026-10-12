"""
SOLD2 Predictor - Inference wrapper for line detection.

Provides a unified API compatible with EdgePredictor for evaluation.
Handles model loading, preprocessing, and post-processing.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from .sold2_network import SOLD2Net, create_sold2_model


class SOLD2Predictor:
    """
    SOLD2 line detector with EdgePredictor-compatible API.
    
    Usage:
        predictor = SOLD2Predictor(checkpoint_path='sold2_model.pt')
        edge_map = predictor.predict(keypoints, img_shape)
        
        # Or with image input:
        lines = predictor.detect_from_image(image)
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
                 descriptor_dim: int = 128):
        """
        Initialize SOLD2 predictor.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: 'cuda', 'cpu', or None for auto-detection
            junction_thresh: Threshold for junction detection
            line_thresh: Threshold for line heatmap
            nms_radius: Radius for non-maximum suppression
            min_line_length: Minimum line segment length in pixels
            max_num_lines: Maximum number of lines to return
            base_channels: Base channels for the network (16 for lightweight, 64 for full)
            descriptor_dim: Descriptor dimension (32 for lightweight, 128 for full)
        """
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Store dimensions
        self.descriptor_dim = descriptor_dim
        
        # Model setup
        self.model = SOLD2Net(in_channels=1, base_channels=base_channels, descriptor_dim=descriptor_dim)
        
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
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
        Predict edge map - compatible with EdgePredictor API.
        
        Args:
            keypoints: (N, 2) array of keypoint coordinates (x, y) - ignored if image provided
            img_shape: (H, W) output shape
            scores: Keypoint scores - unused, for API compatibility
            descriptors: Keypoint descriptors - unused, for API compatibility
            image: Optional input image for direct detection
            
        Returns:
            Binary edge map (H, W) with values 0 or 255
        """
        H, W = img_shape[:2]
        
        # If image provided, use direct detection
        if image is not None:
            return self.detect_edge_map(image)
        
        # Otherwise, create synthetic image from keypoints and detect
        # This is a fallback for API compatibility
        edge_map = np.zeros((H, W), dtype=np.uint8)
        
        if len(keypoints) < 2:
            return edge_map
        
        # Create a sparse image from keypoints
        synth_img = np.zeros((H, W), dtype=np.uint8)
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(synth_img, (x, y), 2, 255, -1)
        
        # Detect lines
        lines = self.detect_from_image(synth_img)
        
        # Draw lines on edge map
        for line in lines:
            y1, x1, y2, x2 = line
            cv2.line(edge_map, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
        
        return edge_map
    
    def detect_edge_map(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges from image and return binary edge map.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            
        Returns:
            Binary edge map (H, W) with values 0 or 255
        """
        # Preprocess
        image_tensor = self._preprocess(image)
        H, W = image.shape[:2]
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Get line heatmap
        line_heatmap = outputs['line_heatmap'][0, 0].cpu().numpy()
        
        # Threshold to binary
        edge_map = (line_heatmap > self.line_thresh).astype(np.uint8) * 255
        
        # Resize to original size if needed
        if edge_map.shape != (H, W):
            edge_map = cv2.resize(edge_map, (W, H), interpolation=cv2.INTER_NEAREST)
        
        return edge_map
    
    def detect_from_image(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect line segments from image.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            
        Returns:
            List of (y1, x1, y2, x2) line segments
        """
        # Preprocess
        image_tensor = self._preprocess(image)
        H_orig, W_orig = image.shape[:2]
        _, _, H_net, W_net = image_tensor.shape
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
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
    
    def get_line_descriptors(self,
                             image: np.ndarray,
                             lines: List[Tuple[int, int, int, int]],
                             num_samples: int = 10) -> np.ndarray:
        """
        Get descriptors for line segments.
        
        Args:
            image: Input image
            lines: List of (y1, x1, y2, x2) line segments
            num_samples: Number of samples per line
            
        Returns:
            (N, D) array of line descriptors
        """
        if len(lines) == 0:
            return np.zeros((0, self.descriptor_dim))
        
        # Preprocess
        image_tensor = self._preprocess(image)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        descriptors = outputs['descriptors'][0]  # (D, H, W)
        D, H, W = descriptors.shape
        
        # Sample descriptors along each line
        line_descs = []
        for y1, x1, y2, x2 in lines:
            # Sample points along line
            t = torch.linspace(0, 1, num_samples)
            ys = (y1 + t * (y2 - y1)).long().clamp(0, H - 1)
            xs = (x1 + t * (x2 - x1)).long().clamp(0, W - 1)
            
            # Get descriptors and average
            sampled = descriptors[:, ys, xs]  # (D, num_samples)
            line_desc = sampled.mean(dim=1)  # (D,)
            line_desc = F.normalize(line_desc, p=2, dim=0)
            
            line_descs.append(line_desc.cpu().numpy())
        
        return np.stack(line_descs, axis=0)
    
    def match_lines(self,
                    descs1: np.ndarray,
                    descs2: np.ndarray,
                    threshold: float = 0.7) -> List[Tuple[int, int, float]]:
        """
        Match lines between two sets using descriptors.
        
        Args:
            descs1: (N, D) descriptors from first image
            descs2: (M, D) descriptors from second image
            threshold: Minimum similarity threshold
            
        Returns:
            List of (idx1, idx2, similarity) matches
        """
        if len(descs1) == 0 or len(descs2) == 0:
            return []
        
        # Compute similarity matrix
        descs1_t = torch.from_numpy(descs1)
        descs2_t = torch.from_numpy(descs2)
        
        similarity = torch.mm(descs1_t, descs2_t.t())  # (N, M)
        
        # Mutual nearest neighbors
        matches = []
        max_sim1, max_idx1 = similarity.max(dim=1)  # Best match for each in set 1
        max_sim2, max_idx2 = similarity.max(dim=0)  # Best match for each in set 2
        
        for i in range(len(descs1)):
            j = max_idx1[i].item()
            if max_idx2[j].item() == i:  # Mutual match
                sim = max_sim1[i].item()
                if sim > threshold:
                    matches.append((i, j, sim))
        
        return matches
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            
        Returns:
            Preprocessed tensor (1, 1, H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Normalize to [0, 1]
        if gray.max() > 1:
            gray = gray.astype(np.float32) / 255.0
        
        # Resize to multiple of 32 for network compatibility
        H, W = gray.shape
        new_H = ((H + 31) // 32) * 32
        new_W = ((W + 31) // 32) * 32
        
        if (H, W) != (new_H, new_W):
            gray = cv2.resize(gray, (new_W, new_H))
        
        # Convert to tensor
        tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _extract_junctions(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Extract junction points from heatmap using NMS.
        
        Args:
            heatmap: (H, W) junction probability map
            
        Returns:
            (N, 2) tensor of (y, x) coordinates
        """
        H, W = heatmap.shape
        pad = self.nms_radius
        
        # Pad and apply max pooling
        heatmap_pad = F.pad(heatmap.unsqueeze(0).unsqueeze(0),
                           (pad, pad, pad, pad), mode='constant', value=0)
        max_pool = F.max_pool2d(heatmap_pad, kernel_size=2*pad+1, stride=1, padding=0)
        max_pool = max_pool.squeeze()
        
        # Find local maxima above threshold
        is_peak = (heatmap == max_pool) & (heatmap > self.junction_thresh)
        
        return torch.nonzero(is_peak, as_tuple=False)
    
    def _match_junctions(self,
                         junctions: torch.Tensor,
                         line_heatmap: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        """
        Match junction pairs to form line segments.
        
        Args:
            junctions: (N, 2) tensor of (y, x) coordinates
            line_heatmap: (H, W) line probability map
            
        Returns:
            List of (y1, x1, y2, x2) line segments
        """
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
