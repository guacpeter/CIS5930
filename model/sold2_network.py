"""
SOLD2 Network Architecture - Unsupervised Line Detection and Description.

This module implements the core neural network components for SOLD2:
- Hourglass backbone for multi-scale feature extraction
- Junction heatmap head for endpoint detection
- Line heatmap head for line segment detection
- Descriptor head for line matching

Based on: "SOLD2: Self-supervised Occlusion-aware Line Description and Detection"
(Lin et al., CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class DownBlock(nn.Module):
    """Downsampling block: Conv + MaxPool."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.residual = ResidualBlock(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.residual(x)
        return self.pool(x), x  # Return pooled and skip connection


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # After concat: in_channels + skip_channels
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        self.residual = ResidualBlock(out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.residual(x)


class HourglassBackbone(nn.Module):
    """
    Hourglass encoder-decoder backbone for feature extraction.
    
    Architecture:
    - Input stem: 3 -> 64 channels
    - 4 downsampling stages: 64 -> 128 -> 256 -> 512
    - 4 upsampling stages with skip connections
    - Output: 64-channel feature map at input resolution
    """
    
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        
        # Input stem
        self.stem = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ResidualBlock(base_channels)
        )
        
        # Encoder (downsampling path)
        self.down1 = DownBlock(base_channels, base_channels * 2)      # 64 -> 128
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)  # 128 -> 256
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)  # 256 -> 512
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )
        
        # Decoder (upsampling path)
        # Skip connections have the OUTPUT channels from each DownBlock:
        # - down1: 64 -> 128, so skip1 = 128 channels
        # - down2: 128 -> 256, so skip2 = 256 channels
        # - down3: 256 -> 512, so skip3 = 512 channels
        # up1: bottleneck=512 + skip3=512 -> 256
        self.up1 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4)
        # up2: 256 + skip2=256 -> 128
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        # up3: 128 + skip1=128 -> 64
        self.up3 = UpBlock(base_channels * 2, base_channels * 2, base_channels)
        
        # Final projection
        self.output_proj = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            ResidualBlock(base_channels)
        )
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        
        # Output
        return self.output_proj(x)


class JunctionHead(nn.Module):
    """
    Junction detection head.
    
    Predicts a heatmap of junction (line endpoint) probability.
    Output: (B, 1, H, W) sigmoid-normalized heatmap.
    """
    
    def __init__(self, in_channels=64):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 2)
        self.conv2 = ConvBlock(in_channels // 2, in_channels // 4)
        self.output = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
    
    def forward(self, features):
        x = self.conv1(features)
        x = self.conv2(x)
        return torch.sigmoid(self.output(x))


class LineHeatmapHead(nn.Module):
    """
    Line heatmap detection head.
    
    Predicts a heatmap of line segment probability.
    Output: (B, 1, H, W) sigmoid-normalized heatmap.
    """
    
    def __init__(self, in_channels=64):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels)
        self.conv2 = ConvBlock(in_channels, in_channels // 2)
        self.conv3 = ConvBlock(in_channels // 2, in_channels // 4)
        self.output = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
    
    def forward(self, features):
        x = self.conv1(features)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.output(x))


class DescriptorHead(nn.Module):
    """
    Line descriptor head.
    
    Generates dense descriptors for each pixel location.
    Output: (B, D, H, W) L2-normalized descriptors.
    """
    
    def __init__(self, in_channels=64, descriptor_dim=128):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels)
        self.conv2 = ConvBlock(in_channels, descriptor_dim)
        self.output = nn.Conv2d(descriptor_dim, descriptor_dim, kernel_size=1)
    
    def forward(self, features):
        x = self.conv1(features)
        x = self.conv2(x)
        x = self.output(x)
        # L2 normalize along descriptor dimension
        return F.normalize(x, p=2, dim=1)


class SOLD2Net(nn.Module):
    """
    Full SOLD2 Network for line detection and description.
    
    Combines:
    - Hourglass backbone for feature extraction
    - Junction head for endpoint detection
    - Line heatmap head for line detection
    - Descriptor head for line matching
    
    Args:
        in_channels: Number of input image channels (1 for grayscale, 3 for RGB)
        base_channels: Base channel count for backbone
        descriptor_dim: Dimension of output descriptors
    """
    
    def __init__(self, in_channels=1, base_channels=64, descriptor_dim=128):
        super().__init__()
        
        self.backbone = HourglassBackbone(in_channels, base_channels)
        self.junction_head = JunctionHead(base_channels)
        self.line_heatmap_head = LineHeatmapHead(base_channels)
        self.descriptor_head = DescriptorHead(base_channels, descriptor_dim)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            dict with:
                - 'junctions': Junction heatmap (B, 1, H, W)
                - 'line_heatmap': Line heatmap (B, 1, H, W)
                - 'descriptors': Dense descriptors (B, D, H, W)
        """
        features = self.backbone(x)
        
        return {
            'junctions': self.junction_head(features),
            'line_heatmap': self.line_heatmap_head(features),
            'descriptors': self.descriptor_head(features)
        }
    
    def detect_lines(self, x, junction_thresh=0.5, line_thresh=0.5):
        """
        Detect line segments from input image.
        
        Args:
            x: Input image tensor (B, C, H, W)
            junction_thresh: Threshold for junction detection
            line_thresh: Threshold for line heatmap
            
        Returns:
            List of detected line segments per batch item
        """
        outputs = self.forward(x)
        
        batch_lines = []
        B = x.shape[0]
        
        for b in range(B):
            junc_map = outputs['junctions'][b, 0]
            line_map = outputs['line_heatmap'][b, 0]
            
            # Extract junction points
            junctions = self._extract_junctions(junc_map, junction_thresh)
            
            # Match junctions to form lines using line heatmap
            lines = self._match_junctions(junctions, line_map, line_thresh)
            
            batch_lines.append(lines)
        
        return batch_lines
    
    def _extract_junctions(self, heatmap, threshold=0.5, nms_radius=4):
        """
        Extract junction points from heatmap using NMS.
        
        Args:
            heatmap: (H, W) tensor of junction probabilities
            threshold: Detection threshold
            nms_radius: Non-maximum suppression radius
            
        Returns:
            (N, 2) tensor of junction coordinates (y, x)
        """
        # Simple peak detection with local maxima
        H, W = heatmap.shape
        
        # Pad for NMS
        pad = nms_radius
        heatmap_pad = F.pad(heatmap.unsqueeze(0).unsqueeze(0), 
                           (pad, pad, pad, pad), mode='constant', value=0)
        
        # Max pooling for NMS
        max_pool = F.max_pool2d(heatmap_pad, kernel_size=2*pad+1, stride=1, padding=0)
        max_pool = max_pool.squeeze()
        
        # Find local maxima above threshold
        is_peak = (heatmap == max_pool) & (heatmap > threshold)
        
        # Get coordinates
        coords = torch.nonzero(is_peak, as_tuple=False)  # (N, 2) as (y, x)
        
        return coords
    
    def _match_junctions(self, junctions, line_heatmap, threshold=0.5):
        """
        Match junction pairs to form line segments.
        
        Args:
            junctions: (N, 2) tensor of junction coordinates
            line_heatmap: (H, W) line probability map
            threshold: Minimum line probability for connection
            
        Returns:
            List of (y1, x1, y2, x2) line segments
        """
        lines = []
        n = len(junctions)
        
        if n < 2:
            return lines
        
        # Check pairs of junctions
        for i in range(n):
            for j in range(i + 1, n):
                p1 = junctions[i].float()
                p2 = junctions[j].float()
                
                # Sample points along the line
                num_samples = max(int(torch.norm(p2 - p1).item()), 10)
                t = torch.linspace(0, 1, num_samples, device=p1.device)
                
                points_y = (p1[0] + t * (p2[0] - p1[0])).long()
                points_x = (p1[1] + t * (p2[1] - p1[1])).long()
                
                # Clamp to valid range
                H, W = line_heatmap.shape
                points_y = points_y.clamp(0, H - 1)
                points_x = points_x.clamp(0, W - 1)
                
                # Check average line probability
                line_probs = line_heatmap[points_y, points_x]
                avg_prob = line_probs.mean().item()
                
                if avg_prob > threshold:
                    y1, x1 = junctions[i].tolist()
                    y2, x2 = junctions[j].tolist()
                    lines.append((y1, x1, y2, x2))
        
        return lines


def create_sold2_model(pretrained=False, checkpoint_path=None, **kwargs):
    """
    Factory function to create SOLD2 model.
    
    Args:
        pretrained: Whether to load pretrained weights
        checkpoint_path: Path to checkpoint file
        **kwargs: Additional arguments for SOLD2Net
        
    Returns:
        SOLD2Net model
    """
    model = SOLD2Net(**kwargs)
    
    if pretrained and checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    return model


class KeypointConditionedSOLD2(nn.Module):
    """
    SOLD2 Network conditioned on SuperPoint keypoints.
    
    This variant takes SuperPoint keypoints as additional input, creating
    a keypoint heatmap that guides the line detection process. The network
    learns to connect keypoints with line segments.
    
    Input: 
        - Image (B, 1, H, W)
        - Keypoint map (B, 1, H, W) - Gaussian blobs at keypoint locations
    
    The two inputs are concatenated along the channel dimension.
    """
    
    def __init__(self, base_channels=64, descriptor_dim=128, keypoint_sigma=4.0):
        """
        Initialize keypoint-conditioned SOLD2.
        
        Args:
            base_channels: Base channel count for backbone
            descriptor_dim: Dimension of output descriptors
            keypoint_sigma: Sigma for Gaussian keypoint blobs
        """
        super().__init__()
        
        self.keypoint_sigma = keypoint_sigma
        
        # Backbone takes 2 channels: image + keypoint map
        self.backbone = HourglassBackbone(in_channels=2, base_channels=base_channels)
        self.junction_head = JunctionHead(base_channels)
        self.line_heatmap_head = LineHeatmapHead(base_channels)
        self.descriptor_head = DescriptorHead(base_channels, descriptor_dim)
    
    def create_keypoint_map(self, keypoints, scores, img_shape, device='cpu'):
        """
        Create a keypoint heatmap from SuperPoint keypoints.
        
        Args:
            keypoints: (N, 2) array of (x, y) keypoint coordinates
            scores: (N,) array of keypoint confidence scores
            img_shape: (H, W) image dimensions
            device: torch device
            
        Returns:
            (1, 1, H, W) keypoint heatmap tensor
        """
        import numpy as np
        
        H, W = img_shape
        heatmap = torch.zeros(1, 1, H, W, device=device)
        
        if len(keypoints) == 0:
            return heatmap
        
        # Convert to numpy if needed
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        # Create Gaussian blobs at keypoint locations
        sigma = self.keypoint_sigma
        size = int(4 * sigma) + 1
        
        # Pre-compute Gaussian kernel
        x = np.arange(-size, size + 1)
        y = np.arange(-size, size + 1)
        xx, yy = np.meshgrid(x, y)
        gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gaussian = torch.from_numpy(gaussian).float().to(device)
        
        for i, (kp, score) in enumerate(zip(keypoints, scores)):
            x_kp, y_kp = int(kp[0]), int(kp[1])
            
            # Compute bounds for placement
            x_min = max(0, x_kp - size)
            x_max = min(W, x_kp + size + 1)
            y_min = max(0, y_kp - size)
            y_max = min(H, y_kp + size + 1)
            
            # Corresponding kernel region
            kx_min = size - (x_kp - x_min)
            kx_max = size + (x_max - x_kp)
            ky_min = size - (y_kp - y_min)
            ky_max = size + (y_max - y_kp)
            
            # Add weighted Gaussian
            if x_max > x_min and y_max > y_min:
                kernel_region = gaussian[ky_min:ky_max, kx_min:kx_max]
                heatmap[0, 0, y_min:y_max, x_min:x_max] = torch.maximum(
                    heatmap[0, 0, y_min:y_max, x_min:x_max],
                    kernel_region * score
                )
        
        return heatmap
    
    def forward(self, image, keypoint_map=None, keypoints=None, scores=None):
        """
        Forward pass with keypoint conditioning.
        
        Args:
            image: Input image tensor (B, 1, H, W)
            keypoint_map: Pre-computed keypoint heatmap (B, 1, H, W) OR
            keypoints: List of (N, 2) keypoint arrays per batch item
            scores: List of (N,) score arrays per batch item
            
        Returns:
            dict with:
                - 'junctions': Junction heatmap (B, 1, H, W)
                - 'line_heatmap': Line heatmap (B, 1, H, W)
                - 'descriptors': Dense descriptors (B, D, H, W)
        """
        B, C, H, W = image.shape
        device = image.device
        
        # Create keypoint map if not provided
        if keypoint_map is None:
            if keypoints is not None and scores is not None:
                keypoint_maps = []
                for i in range(B):
                    kp = keypoints[i] if isinstance(keypoints, list) else keypoints
                    sc = scores[i] if isinstance(scores, list) else scores
                    kp_map = self.create_keypoint_map(kp, sc, (H, W), device)
                    keypoint_maps.append(kp_map)
                keypoint_map = torch.cat(keypoint_maps, dim=0)
            else:
                # No keypoints - use zeros
                keypoint_map = torch.zeros(B, 1, H, W, device=device)
        
        # Concatenate image and keypoint map
        x = torch.cat([image, keypoint_map], dim=1)  # (B, 2, H, W)
        
        # Forward through network
        features = self.backbone(x)
        
        return {
            'junctions': self.junction_head(features),
            'line_heatmap': self.line_heatmap_head(features),
            'descriptors': self.descriptor_head(features),
            'keypoint_map': keypoint_map  # Return for visualization
        }


def create_keypoint_sold2_model(pretrained=False, checkpoint_path=None, **kwargs):
    """
    Factory function to create keypoint-conditioned SOLD2 model.
    
    Args:
        pretrained: Whether to load pretrained weights
        checkpoint_path: Path to checkpoint file
        **kwargs: Additional arguments for KeypointConditionedSOLD2
        
    Returns:
        KeypointConditionedSOLD2 model
    """
    model = KeypointConditionedSOLD2(**kwargs)
    
    if pretrained and checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    return model
