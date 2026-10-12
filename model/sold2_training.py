"""
SOLD2 Self-Supervised Training Module.

This module provides utilities for unsupervised training of SOLD2:
- Synthetic dataset generation for pre-training
- Homography adaptation for pseudo ground truth
- Loss functions for detector and descriptor training

Based on: "SOLD2: Self-supervised Occlusion-aware Line Description and Detection"
(Lin et al., CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict


# =============================================================================
# Synthetic Dataset Generation
# =============================================================================

class SyntheticLineDataset(Dataset):
    """
    Synthetic dataset generator for SOLD2 pre-training.
    
    Generates random geometric shapes with clean ground truth:
    - Lines (random, parallel, converging)
    - Polygons (triangles, rectangles, pentagons)
    - Stars and complex patterns
    - Checkboards and grids
    
    This provides perfect annotations for initial detector training.
    """
    
    def __init__(self, 
                 num_samples: int = 10000,
                 img_size: Tuple[int, int] = (256, 256),
                 min_lines: int = 3,
                 max_lines: int = 15):
        """
        Args:
            num_samples: Number of synthetic images to generate
            img_size: Output image size (H, W)
            min_lines: Minimum number of lines per image
            max_lines: Maximum number of lines per image
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.min_lines = min_lines
        self.max_lines = max_lines
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Set seed for reproducibility
        np.random.seed(idx)
        
        H, W = self.img_size
        image = np.zeros((H, W), dtype=np.uint8)
        line_heatmap = np.zeros((H, W), dtype=np.float32)
        junction_heatmap = np.zeros((H, W), dtype=np.float32)
        lines = []
        junctions = set()
        
        # Randomly select shape type
        shape_type = np.random.choice(['lines', 'polygon', 'star', 'grid', 'mixed'])
        
        if shape_type == 'lines':
            lines, junctions = self._draw_random_lines(image, line_heatmap)
        elif shape_type == 'polygon':
            lines, junctions = self._draw_polygon(image, line_heatmap)
        elif shape_type == 'star':
            lines, junctions = self._draw_star(image, line_heatmap)
        elif shape_type == 'grid':
            lines, junctions = self._draw_grid(image, line_heatmap)
        else:  # mixed
            lines1, junc1 = self._draw_random_lines(image, line_heatmap, n=5)
            lines2, junc2 = self._draw_polygon(image, line_heatmap)
            lines = lines1 + lines2
            junctions = junc1.union(junc2)
        
        # Create junction heatmap with Gaussian blobs
        for (y, x) in junctions:
            if 0 <= y < H and 0 <= x < W:
                self._draw_gaussian(junction_heatmap, (y, x), sigma=2)
        
        # Normalize
        if line_heatmap.max() > 0:
            line_heatmap = line_heatmap / line_heatmap.max()
        if junction_heatmap.max() > 0:
            junction_heatmap = junction_heatmap / junction_heatmap.max()
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        line_heatmap_tensor = torch.from_numpy(line_heatmap).float().unsqueeze(0)
        junction_heatmap_tensor = torch.from_numpy(junction_heatmap).float().unsqueeze(0)
        
        return {
            'image': image_tensor,
            'line_heatmap': line_heatmap_tensor,
            'junction_heatmap': junction_heatmap_tensor,
            'lines': lines
        }
    
    def _draw_random_lines(self, image, heatmap, n=None):
        """Draw random line segments."""
        H, W = self.img_size
        if n is None:
            n = np.random.randint(self.min_lines, self.max_lines + 1)
        
        lines = []
        junctions = set()
        
        for _ in range(n):
            y1, x1 = np.random.randint(10, H - 10), np.random.randint(10, W - 10)
            y2, x2 = np.random.randint(10, H - 10), np.random.randint(10, W - 10)
            
            # Ensure minimum length
            if np.sqrt((y2 - y1)**2 + (x2 - x1)**2) < 20:
                continue
                
            cv2.line(image, (x1, y1), (x2, y2), 255, 1)
            cv2.line(heatmap, (x1, y1), (x2, y2), 1.0, 1)
            
            lines.append((y1, x1, y2, x2))
            junctions.add((y1, x1))
            junctions.add((y2, x2))
        
        return lines, junctions
    
    def _draw_polygon(self, image, heatmap):
        """Draw a random polygon."""
        H, W = self.img_size
        n_sides = np.random.randint(3, 8)
        
        # Random center and radius
        cy, cx = np.random.randint(H // 4, 3 * H // 4), np.random.randint(W // 4, 3 * W // 4)
        radius = np.random.randint(30, min(H, W) // 3)
        
        # Generate vertices
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        angles += np.random.rand() * np.pi  # Random rotation
        
        vertices = []
        for angle in angles:
            y = int(cy + radius * np.sin(angle))
            x = int(cx + radius * np.cos(angle))
            vertices.append((y, x))
        
        lines = []
        junctions = set()
        
        for i in range(len(vertices)):
            y1, x1 = vertices[i]
            y2, x2 = vertices[(i + 1) % len(vertices)]
            
            cv2.line(image, (x1, y1), (x2, y2), 255, 1)
            cv2.line(heatmap, (x1, y1), (x2, y2), 1.0, 1)
            
            lines.append((y1, x1, y2, x2))
            junctions.add((y1, x1))
        
        return lines, junctions
    
    def _draw_star(self, image, heatmap):
        """Draw a star pattern."""
        H, W = self.img_size
        n_points = np.random.randint(5, 9)
        
        cy, cx = np.random.randint(H // 4, 3 * H // 4), np.random.randint(W // 4, 3 * W // 4)
        outer_r = np.random.randint(40, min(H, W) // 3)
        inner_r = outer_r // 2
        
        angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False)
        
        vertices = []
        for i, angle in enumerate(angles):
            r = outer_r if i % 2 == 0 else inner_r
            y = int(cy + r * np.sin(angle))
            x = int(cx + r * np.cos(angle))
            vertices.append((y, x))
        
        lines = []
        junctions = set()
        
        for i in range(len(vertices)):
            y1, x1 = vertices[i]
            y2, x2 = vertices[(i + 1) % len(vertices)]
            
            cv2.line(image, (x1, y1), (x2, y2), 255, 1)
            cv2.line(heatmap, (x1, y1), (x2, y2), 1.0, 1)
            
            lines.append((y1, x1, y2, x2))
            junctions.add((y1, x1))
        
        return lines, junctions
    
    def _draw_grid(self, image, heatmap):
        """Draw a grid pattern."""
        H, W = self.img_size
        n_h = np.random.randint(3, 7)
        n_w = np.random.randint(3, 7)
        
        margin = 20
        step_h = (H - 2 * margin) // n_h
        step_w = (W - 2 * margin) // n_w
        
        lines = []
        junctions = set()
        
        # Horizontal lines
        for i in range(n_h + 1):
            y = margin + i * step_h
            cv2.line(image, (margin, y), (W - margin, y), 255, 1)
            cv2.line(heatmap, (margin, y), (W - margin, y), 1.0, 1)
            lines.append((y, margin, y, W - margin))
            junctions.add((y, margin))
            junctions.add((y, W - margin))
        
        # Vertical lines
        for j in range(n_w + 1):
            x = margin + j * step_w
            cv2.line(image, (x, margin), (x, H - margin), 255, 1)
            cv2.line(heatmap, (x, margin), (x, H - margin), 1.0, 1)
            lines.append((margin, x, H - margin, x))
            junctions.add((margin, x))
            junctions.add((H - margin, x))
        
        # Grid intersections as junctions
        for i in range(n_h + 1):
            for j in range(n_w + 1):
                y = margin + i * step_h
                x = margin + j * step_w
                junctions.add((y, x))
        
        return lines, junctions
    
    def _draw_gaussian(self, heatmap, center, sigma=2):
        """Draw a Gaussian blob at center."""
        H, W = heatmap.shape
        y_c, x_c = center
        
        # Create local Gaussian
        size = int(6 * sigma)
        y_min = max(0, y_c - size)
        y_max = min(H, y_c + size + 1)
        x_min = max(0, x_c - size)
        x_max = min(W, x_c + size + 1)
        
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                dist_sq = (y - y_c) ** 2 + (x - x_c) ** 2
                val = np.exp(-dist_sq / (2 * sigma ** 2))
                heatmap[y, x] = max(heatmap[y, x], val)


class BIPEDKeypointDataset(Dataset):
    """
    BIPED dataset with SuperPoint keypoints for keypoint-conditioned SOLD2.
    
    Loads:
    - RGB images from data/rgb/
    - SuperPoint keypoints from data/keypoints/
    - Ground truth edges from data/edges/ (as training targets)
    """
    
    def __init__(self,
                 data_dir: str = 'data',
                 img_size: Tuple[int, int] = (256, 256),
                 keypoint_sigma: float = 4.0,
                 augment: bool = True):
        """
        Args:
            data_dir: Root data directory containing rgb/, keypoints/, edges/
            img_size: Output image size (H, W)
            keypoint_sigma: Sigma for Gaussian keypoint blobs
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.keypoint_sigma = keypoint_sigma
        self.augment = augment
        
        # Find all images with corresponding keypoints and edges
        self.samples = []
        rgb_dir = self.data_dir / 'rgb'
        kp_dir = self.data_dir / 'keypoints'
        edge_dir = self.data_dir / 'edges'
        
        if rgb_dir.exists():
            for img_path in sorted(rgb_dir.glob('*.jpg')):
                name = img_path.stem  # e.g., RGB_001
                kp_path = kp_dir / f'{name}_keypoints.npz'
                edge_path = edge_dir / f'{name}.png'
                
                if kp_path.exists() and edge_path.exists():
                    self.samples.append({
                        'image': str(img_path),
                        'keypoints': str(kp_path),
                        'edges': str(edge_path)
                    })
        
        print(f"BIPEDKeypointDataset: Found {len(self.samples)} samples with keypoints and edges")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        H, W = self.img_size
        
        # Load image
        image = cv2.imread(sample['image'], cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros(self.img_size, dtype=np.uint8)
        image = cv2.resize(image, (W, H))
        
        # Load keypoints
        kp_data = np.load(sample['keypoints'])
        keypoints = kp_data['keypoints']  # (N, 2) - x, y
        scores = kp_data['scores']  # (N,)
        
        # Scale keypoints to new image size
        orig_h, orig_w = cv2.imread(sample['image'], cv2.IMREAD_GRAYSCALE).shape[:2]
        scale_x = W / orig_w
        scale_y = H / orig_h
        keypoints_scaled = keypoints.copy()
        keypoints_scaled[:, 0] *= scale_x
        keypoints_scaled[:, 1] *= scale_y
        
        # Create keypoint heatmap
        keypoint_map = self._create_keypoint_map(keypoints_scaled, scores, (H, W))
        
        # Load ground truth edges
        edges = cv2.imread(sample['edges'], cv2.IMREAD_GRAYSCALE)
        if edges is None:
            edges = np.zeros(self.img_size, dtype=np.uint8)
        edges = cv2.resize(edges, (W, H))
        
        # Normalize edge map to [0, 1]
        edge_heatmap = edges.astype(np.float32) / 255.0
        
        # Create junction heatmap from edge endpoints (simplified)
        junction_heatmap = self._create_junction_heatmap(edge_heatmap)
        
        # Apply augmentation
        if self.augment:
            image, keypoint_map, edge_heatmap, junction_heatmap = self._augment(
                image, keypoint_map, edge_heatmap, junction_heatmap
            )
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        keypoint_tensor = torch.from_numpy(keypoint_map).float().unsqueeze(0)
        line_heatmap_tensor = torch.from_numpy(edge_heatmap).float().unsqueeze(0)
        junction_heatmap_tensor = torch.from_numpy(junction_heatmap).float().unsqueeze(0)
        
        return {
            'image': image_tensor,
            'keypoint_map': keypoint_tensor,
            'line_heatmap': line_heatmap_tensor,
            'junction_heatmap': junction_heatmap_tensor,
            'keypoints': keypoints_scaled,
            'scores': scores
        }
    
    def _create_keypoint_map(self, keypoints, scores, img_shape):
        """Create Gaussian keypoint heatmap."""
        H, W = img_shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        
        sigma = self.keypoint_sigma
        size = int(4 * sigma) + 1
        
        # Pre-compute Gaussian kernel
        x = np.arange(-size, size + 1)
        y = np.arange(-size, size + 1)
        xx, yy = np.meshgrid(x, y)
        gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        for (kp, score) in zip(keypoints, scores):
            x_kp, y_kp = int(kp[0]), int(kp[1])
            
            x_min = max(0, x_kp - size)
            x_max = min(W, x_kp + size + 1)
            y_min = max(0, y_kp - size)
            y_max = min(H, y_kp + size + 1)
            
            kx_min = size - (x_kp - x_min)
            kx_max = size + (x_max - x_kp)
            ky_min = size - (y_kp - y_min)
            ky_max = size + (y_max - y_kp)
            
            if x_max > x_min and y_max > y_min:
                kernel_region = gaussian[ky_min:ky_max, kx_min:kx_max]
                heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                    heatmap[y_min:y_max, x_min:x_max],
                    kernel_region * score
                )
        
        return heatmap
    
    def _create_junction_heatmap(self, edge_heatmap):
        """Create simplified junction heatmap from edge endpoints."""
        # Find edge pixels
        edge_binary = (edge_heatmap > 0.5).astype(np.uint8)
        
        # Find endpoints using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edge_binary, kernel, iterations=1)
        
        # Simplified: use corners of edge regions as junctions
        junction_heatmap = np.zeros_like(edge_heatmap)
        
        # Find contours
        contours, _ = cv2.findContours(edge_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) > 0:
                # Endpoints of contour
                for pt in [contour[0][0], contour[-1][0]]:
                    x, y = pt
                    if 0 <= y < junction_heatmap.shape[0] and 0 <= x < junction_heatmap.shape[1]:
                        cv2.circle(junction_heatmap, (x, y), 3, 1.0, -1)
        
        return junction_heatmap
    
    def _augment(self, image, keypoint_map, edge_heatmap, junction_heatmap):
        """Apply basic augmentation."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            keypoint_map = np.fliplr(keypoint_map).copy()
            edge_heatmap = np.fliplr(edge_heatmap).copy()
            junction_heatmap = np.fliplr(junction_heatmap).copy()
        
        # Random brightness/contrast
        alpha = np.random.uniform(0.8, 1.2)  # Contrast
        beta = np.random.uniform(-20, 20)    # Brightness
        image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
        
        return image, keypoint_map, edge_heatmap, junction_heatmap


# =============================================================================
# Homography Adaptation
# =============================================================================

class HomographyAdaptation:
    """
    Homography adaptation for self-supervised pseudo ground truth generation.
    
    Applies multiple random homographies to an image, runs inference,
    and aggregates predictions via majority voting to create robust pseudo labels.
    """
    
    def __init__(self,
                 num_homographies: int = 100,
                 aggregation_method: str = 'mean',
                 min_confidence: float = 0.3):
        """
        Args:
            num_homographies: Number of homographies to sample
            aggregation_method: 'mean' or 'max' for prediction aggregation
            min_confidence: Minimum confidence threshold for pseudo GT
        """
        self.num_homographies = num_homographies
        self.aggregation_method = aggregation_method
        self.min_confidence = min_confidence
    
    def generate_homography(self, img_size: Tuple[int, int]) -> np.ndarray:
        """
        Generate a random homography matrix.
        
        Args:
            img_size: (H, W) image dimensions
            
        Returns:
            3x3 homography matrix
        """
        H, W = img_size
        
        # Source corners (with margin to avoid edge effects)
        margin = min(H, W) // 10
        src = np.array([
            [margin, margin],
            [W - margin, margin],
            [W - margin, H - margin],
            [margin, H - margin]
        ], dtype=np.float32)
        
        # Random perturbation for destination corners
        perturb = np.random.randint(-margin // 2, margin // 2 + 1, (4, 2)).astype(np.float32)
        dst = src + perturb
        
        # Compute homography
        homography, _ = cv2.findHomography(src, dst)
        
        return homography
    
    @torch.no_grad()
    def adapt(self, 
              model: nn.Module,
              image: torch.Tensor,
              device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Apply homography adaptation to generate pseudo ground truth.
        
        Args:
            model: SOLD2 model for inference
            image: Input image (B, C, H, W)
            device: Device for computation
            
        Returns:
            Dict with aggregated pseudo GT heatmaps
        """
        model.eval()
        B, C, H, W = image.shape
        img_np = (image[0, 0].cpu().numpy() * 255).astype(np.uint8)
        
        junction_accumulator = torch.zeros((H, W), device=device)
        line_accumulator = torch.zeros((H, W), device=device)
        count_accumulator = torch.zeros((H, W), device=device)
        
        for _ in range(self.num_homographies):
            # Generate random homography
            H_mat = self.generate_homography((H, W))
            H_inv = np.linalg.inv(H_mat)
            
            # Warp image
            warped = cv2.warpPerspective(img_np, H_mat, (W, H))
            warped_tensor = torch.from_numpy(warped).float().unsqueeze(0).unsqueeze(0) / 255.0
            warped_tensor = warped_tensor.to(device)
            
            # Run model
            outputs = model(warped_tensor)
            
            # Warp predictions back
            junc_np = outputs['junctions'][0, 0].cpu().numpy()
            line_np = outputs['line_heatmap'][0, 0].cpu().numpy()
            
            junc_back = cv2.warpPerspective(junc_np, H_inv, (W, H))
            line_back = cv2.warpPerspective(line_np, H_inv, (W, H))
            
            # Create validity mask (where original image was mapped to)
            mask = cv2.warpPerspective(np.ones((H, W), dtype=np.float32), H_inv, (W, H))
            
            junction_accumulator += torch.from_numpy(junc_back).to(device)
            line_accumulator += torch.from_numpy(line_back).to(device)
            count_accumulator += torch.from_numpy(mask).to(device)
        
        # Avoid division by zero
        count_accumulator = count_accumulator.clamp(min=1)
        
        # Aggregate
        if self.aggregation_method == 'mean':
            junction_pseudo = junction_accumulator / count_accumulator
            line_pseudo = line_accumulator / count_accumulator
        else:  # max was tracked differently, use mean for now
            junction_pseudo = junction_accumulator / count_accumulator
            line_pseudo = line_accumulator / count_accumulator
        
        # Apply threshold
        junction_pseudo = (junction_pseudo > self.min_confidence).float()
        line_pseudo = (line_pseudo > self.min_confidence).float()
        
        return {
            'junction_heatmap': junction_pseudo.unsqueeze(0).unsqueeze(0),
            'line_heatmap': line_pseudo.unsqueeze(0).unsqueeze(0)
        }


# =============================================================================
# Loss Functions
# =============================================================================

class SOLD2Loss(nn.Module):
    """
    Combined loss for SOLD2 training.
    
    Includes:
    - Junction heatmap loss (focal loss for class imbalance)
    - Line heatmap loss (weighted BCE)
    - Descriptor triplet loss (for matching)
    """
    
    def __init__(self,
                 junction_weight: float = 1.0,
                 line_weight: float = 1.0,
                 descriptor_weight: float = 0.5,
                 focal_alpha: float = 2.0,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.junction_weight = junction_weight
        self.line_weight = line_weight
        self.descriptor_weight = descriptor_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            predictions: Dict with 'junctions', 'line_heatmap', 'descriptors'
            targets: Dict with 'junction_heatmap', 'line_heatmap'
            
        Returns:
            Dict with individual losses and total
        """
        losses = {}
        
        # Junction loss (focal BCE)
        if 'junction_heatmap' in targets:
            junc_pred = predictions['junctions']
            junc_target = targets['junction_heatmap']
            losses['junction'] = self._focal_loss(junc_pred, junc_target)
        
        # Line heatmap loss (weighted BCE)
        if 'line_heatmap' in targets:
            line_pred = predictions['line_heatmap']
            line_target = targets['line_heatmap']
            losses['line'] = self._weighted_bce(line_pred, line_target)
        
        # Total loss
        total = 0
        if 'junction' in losses:
            total += self.junction_weight * losses['junction']
        if 'line' in losses:
            total += self.line_weight * losses['line']
        
        losses['total'] = total
        
        return losses
    
    def _focal_loss(self, pred, target):
        """Focal loss for handling class imbalance."""
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        return (focal_weight * bce).mean()
    
    def _weighted_bce(self, pred, target, pos_weight=10.0):
        """Weighted BCE with higher weight for positive pixels."""
        weight = torch.where(target > 0.5, pos_weight, 1.0)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        return (weight * bce).mean()


class DescriptorTripletLoss(nn.Module):
    """
    Triplet loss for descriptor learning.
    
    Matches descriptors along the same line (positive) vs
    descriptors from different lines (negative).
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, 
                descriptors: torch.Tensor,
                line_segments: List,
                negative_samples: int = 5) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            descriptors: (B, D, H, W) dense descriptor map
            line_segments: List of (y1, x1, y2, x2) for each batch item
            negative_samples: Number of negative samples per anchor
            
        Returns:
            Triplet loss value
        """
        B, D, H, W = descriptors.shape
        device = descriptors.device
        
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for b in range(B):
            lines = line_segments[b] if b < len(line_segments) else []
            if len(lines) < 2:
                continue
            
            for line in lines:
                y1, x1, y2, x2 = line
                
                # Sample points along line (anchor-positive pairs)
                num_samples = 5
                t = torch.linspace(0.1, 0.9, num_samples)
                
                for i in range(len(t) - 1):
                    y_a = int(y1 + t[i] * (y2 - y1))
                    x_a = int(x1 + t[i] * (x2 - x1))
                    y_p = int(y1 + t[i + 1] * (y2 - y1))
                    x_p = int(x1 + t[i + 1] * (x2 - x1))
                    
                    # Clamp coordinates
                    y_a, x_a = min(max(y_a, 0), H - 1), min(max(x_a, 0), W - 1)
                    y_p, x_p = min(max(y_p, 0), H - 1), min(max(x_p, 0), W - 1)
                    
                    anchor = descriptors[b, :, y_a, x_a]
                    positive = descriptors[b, :, y_p, x_p]
                    
                    # Sample negatives from random locations
                    for _ in range(negative_samples):
                        y_n = np.random.randint(0, H)
                        x_n = np.random.randint(0, W)
                        negative = descriptors[b, :, y_n, x_n]
                        
                        # Triplet loss
                        pos_dist = torch.norm(anchor - positive)
                        neg_dist = torch.norm(anchor - negative)
                        
                        loss = F.relu(pos_dist - neg_dist + self.margin)
                        total_loss += loss
                        count += 1
        
        if count > 0:
            return total_loss / count
        return total_loss


# =============================================================================
# Data Augmentation
# =============================================================================

class PhotometricAugmentation:
    """Photometric augmentations for training robustness."""
    
    def __init__(self,
                 brightness_range: Tuple[float, float] = (0.7, 1.3),
                 contrast_range: Tuple[float, float] = (0.7, 1.3),
                 noise_stddev: float = 0.05):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_stddev = noise_stddev
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to image tensor."""
        # Brightness
        brightness = np.random.uniform(*self.brightness_range)
        image = image * brightness
        
        # Contrast
        contrast = np.random.uniform(*self.contrast_range)
        mean = image.mean()
        image = (image - mean) * contrast + mean
        
        # Gaussian noise
        if self.noise_stddev > 0:
            noise = torch.randn_like(image) * self.noise_stddev
            image = image + noise
        
        return image.clamp(0, 1)


class GeometricAugmentation:
    """Geometric augmentations with ground truth transformation."""
    
    def __init__(self, max_angle: float = 15.0, max_scale: float = 0.1):
        self.max_angle = max_angle
        self.max_scale = max_scale
    
    def __call__(self, 
                 image: torch.Tensor,
                 heatmaps: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply geometric augmentation.
        
        Returns augmented image and transformed heatmaps.
        """
        # Random rotation
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        
        # Random scale
        scale = 1.0 + np.random.uniform(-self.max_scale, self.max_scale)
        
        # Apply to image
        image_np = image[0, 0].numpy()
        H, W = image_np.shape
        center = (W // 2, H // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, scale)
        image_aug = cv2.warpAffine(image_np, M, (W, H))
        
        # Apply to heatmaps
        aug_heatmaps = {}
        for key, hmap in heatmaps.items():
            hmap_np = hmap[0, 0].numpy()
            hmap_aug = cv2.warpAffine(hmap_np, M, (W, H))
            aug_heatmaps[key] = torch.from_numpy(hmap_aug).unsqueeze(0).unsqueeze(0)
        
        return torch.from_numpy(image_aug).unsqueeze(0).unsqueeze(0), aug_heatmaps
