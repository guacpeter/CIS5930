"""
Unit tests for SOLD2 modules.

Tests cover:
- Network forward pass and output shapes
- Synthetic dataset generation
- Homography adaptation
- Loss functions
- Predictor functionality
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.sold2_network import (
    SOLD2Net, HourglassBackbone, JunctionHead, 
    LineHeatmapHead, DescriptorHead, create_sold2_model
)
from model.sold2_training import (
    SyntheticLineDataset, HomographyAdaptation, SOLD2Loss
)
from model.sold2_predictor import SOLD2Predictor


# =============================================================================
# Network Tests
# =============================================================================

class TestSOLD2Network:
    """Tests for SOLD2 network architecture."""
    
    @pytest.fixture
    def model(self):
        return SOLD2Net(in_channels=1, base_channels=32, descriptor_dim=64)
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 1, 256, 256)
    
    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'junction_head')
        assert hasattr(model, 'line_heatmap_head')
        assert hasattr(model, 'descriptor_head')
    
    def test_forward_pass(self, model, sample_input):
        """Test forward pass returns expected outputs."""
        outputs = model(sample_input)
        
        assert 'junctions' in outputs
        assert 'line_heatmap' in outputs
        assert 'descriptors' in outputs
    
    def test_output_shapes(self, model, sample_input):
        """Test output tensor shapes."""
        B, C, H, W = sample_input.shape
        outputs = model(sample_input)
        
        # Junction heatmap: (B, 1, H, W)
        assert outputs['junctions'].shape == (B, 1, H, W)
        
        # Line heatmap: (B, 1, H, W)
        assert outputs['line_heatmap'].shape == (B, 1, H, W)
        
        # Descriptors: (B, D, H, W)
        assert outputs['descriptors'].shape == (B, 64, H, W)
    
    def test_output_ranges(self, model, sample_input):
        """Test output values are in expected ranges."""
        outputs = model(sample_input)
        
        # Heatmaps should be in [0, 1] (sigmoid)
        assert outputs['junctions'].min() >= 0
        assert outputs['junctions'].max() <= 1
        assert outputs['line_heatmap'].min() >= 0
        assert outputs['line_heatmap'].max() <= 1
        
        # Descriptors should be L2-normalized (unit norm)
        desc_norms = outputs['descriptors'].norm(dim=1)
        assert torch.allclose(desc_norms, torch.ones_like(desc_norms), atol=1e-5)
    
    def test_backbone_only(self):
        """Test backbone produces features."""
        backbone = HourglassBackbone(in_channels=1, base_channels=32)
        x = torch.randn(2, 1, 128, 128)
        features = backbone(x)
        
        assert features.shape == (2, 32, 128, 128)
    
    def test_detect_lines(self, model, sample_input):
        """Test line detection method."""
        lines = model.detect_lines(sample_input)
        
        assert isinstance(lines, list)
        assert len(lines) == sample_input.shape[0]  # One list per batch item
    
    def test_factory_function(self):
        """Test create_sold2_model factory."""
        model = create_sold2_model(pretrained=False)
        assert isinstance(model, SOLD2Net)


# =============================================================================
# Training Module Tests
# =============================================================================

class TestSyntheticDataset:
    """Tests for synthetic dataset generation."""
    
    @pytest.fixture
    def dataset(self):
        return SyntheticLineDataset(num_samples=10, img_size=(128, 128))
    
    def test_dataset_length(self, dataset):
        """Test dataset has correct length."""
        assert len(dataset) == 10
    
    def test_sample_structure(self, dataset):
        """Test sample has expected keys."""
        sample = dataset[0]
        
        assert 'image' in sample
        assert 'line_heatmap' in sample
        assert 'junction_heatmap' in sample
        assert 'lines' in sample
    
    def test_sample_shapes(self, dataset):
        """Test sample tensors have correct shapes."""
        sample = dataset[0]
        
        assert sample['image'].shape == (1, 128, 128)
        assert sample['line_heatmap'].shape == (1, 128, 128)
        assert sample['junction_heatmap'].shape == (1, 128, 128)
    
    def test_sample_ranges(self, dataset):
        """Test sample values are in valid ranges."""
        sample = dataset[0]
        
        # Image in [0, 1]
        assert sample['image'].min() >= 0
        assert sample['image'].max() <= 1
        
        # Heatmaps in [0, 1]
        assert sample['line_heatmap'].min() >= 0
        assert sample['line_heatmap'].max() <= 1
    
    def test_reproducibility(self, dataset):
        """Test same index gives same sample."""
        sample1 = dataset[5]
        sample2 = dataset[5]
        
        assert torch.allclose(sample1['image'], sample2['image'])
    
    def test_variability(self, dataset):
        """Test different indices give different samples."""
        sample1 = dataset[0]
        sample2 = dataset[1]
        
        assert not torch.allclose(sample1['image'], sample2['image'])


class TestSOLD2Loss:
    """Tests for SOLD2 loss functions."""
    
    @pytest.fixture
    def criterion(self):
        return SOLD2Loss(junction_weight=1.0, line_weight=1.0)
    
    def test_loss_computation(self, criterion):
        """Test loss computes without error."""
        predictions = {
            'junctions': torch.rand(2, 1, 64, 64),
            'line_heatmap': torch.rand(2, 1, 64, 64),
            'descriptors': torch.rand(2, 64, 64, 64)
        }
        targets = {
            'junction_heatmap': torch.rand(2, 1, 64, 64),
            'line_heatmap': torch.rand(2, 1, 64, 64)
        }
        
        losses = criterion(predictions, targets)
        
        assert 'junction' in losses
        assert 'line' in losses
        assert 'total' in losses
    
    def test_loss_positive(self, criterion):
        """Test loss is positive."""
        predictions = {
            'junctions': torch.rand(2, 1, 32, 32),
            'line_heatmap': torch.rand(2, 1, 32, 32)
        }
        targets = {
            'junction_heatmap': torch.rand(2, 1, 32, 32),
            'line_heatmap': torch.rand(2, 1, 32, 32)
        }
        
        losses = criterion(predictions, targets)
        
        assert losses['total'].item() >= 0
    
    def test_loss_gradient(self, criterion):
        """Test loss has gradient."""
        predictions = {
            'junctions': torch.rand(1, 1, 32, 32, requires_grad=True),
            'line_heatmap': torch.rand(1, 1, 32, 32, requires_grad=True)
        }
        targets = {
            'junction_heatmap': torch.rand(1, 1, 32, 32),
            'line_heatmap': torch.rand(1, 1, 32, 32)
        }
        
        losses = criterion(predictions, targets)
        losses['total'].backward()
        
        assert predictions['junctions'].grad is not None
        assert predictions['line_heatmap'].grad is not None


# =============================================================================
# Predictor Tests
# =============================================================================

class TestSOLD2Predictor:
    """Tests for SOLD2 predictor."""
    
    @pytest.fixture
    def predictor(self):
        return SOLD2Predictor(device='cpu')
    
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    def test_predictor_initialization(self, predictor):
        """Test predictor initializes correctly."""
        assert predictor.model is not None
        assert predictor.device == 'cpu'
    
    def test_predict_from_keypoints(self, predictor):
        """Test prediction from keypoints."""
        keypoints = np.array([[50, 50], [100, 100], [150, 50]], dtype=float)
        edge_map = predictor.predict(keypoints, (256, 256))
        
        assert edge_map.shape == (256, 256)
        assert edge_map.dtype == np.uint8
    
    def test_predict_empty_keypoints(self, predictor):
        """Test prediction with empty keypoints."""
        keypoints = np.array([], dtype=float).reshape(0, 2)
        edge_map = predictor.predict(keypoints, (100, 100))
        
        assert edge_map.shape == (100, 100)
        assert np.sum(edge_map) == 0
    
    def test_detect_from_image(self, predictor, sample_image):
        """Test line detection from image."""
        lines = predictor.detect_from_image(sample_image)
        
        assert isinstance(lines, list)
        # Lines should be tuples of (y1, x1, y2, x2)
        for line in lines:
            assert len(line) == 4
    
    def test_detect_edge_map(self, predictor, sample_image):
        """Test edge map detection from image."""
        edge_map = predictor.detect_edge_map(sample_image)
        
        assert edge_map.shape == sample_image.shape
        assert edge_map.dtype == np.uint8
    
    def test_preprocessing(self, predictor, sample_image):
        """Test image preprocessing."""
        tensor = predictor._preprocess(sample_image)
        
        assert tensor.dim() == 4  # (B, C, H, W)
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 1
        assert tensor.min() >= 0
        assert tensor.max() <= 1
    
    def test_rgb_image(self, predictor):
        """Test predictor handles RGB images."""
        rgb_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        edge_map = predictor.detect_edge_map(rgb_image)
        
        assert edge_map.shape == (256, 256)
    
    def test_get_descriptors(self, predictor, sample_image):
        """Test line descriptor extraction."""
        lines = [(50, 50, 100, 100), (30, 30, 80, 80)]
        descs = predictor.get_line_descriptors(sample_image, lines)
        
        assert descs.shape == (2, 128)
        # Descriptors should be normalized
        norms = np.linalg.norm(descs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
    
    def test_match_lines(self, predictor):
        """Test line matching with descriptors."""
        # Create similar descriptors
        descs1 = np.random.randn(5, 128).astype(np.float32)
        descs1 = descs1 / np.linalg.norm(descs1, axis=1, keepdims=True)
        
        # descs2 is slightly perturbed version
        descs2 = descs1 + np.random.randn(5, 128).astype(np.float32) * 0.1
        descs2 = descs2 / np.linalg.norm(descs2, axis=1, keepdims=True)
        
        matches = predictor.match_lines(descs1, descs2, threshold=0.5)
        
        assert isinstance(matches, list)
        # Should find some matches
        assert len(matches) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_synthetic_to_model(self):
        """Test synthetic data flows through model."""
        dataset = SyntheticLineDataset(num_samples=2, img_size=(128, 128))
        model = SOLD2Net(in_channels=1, base_channels=32, descriptor_dim=64)
        criterion = SOLD2Loss()
        
        sample = dataset[0]
        image = sample['image'].unsqueeze(0)
        targets = {
            'junction_heatmap': sample['junction_heatmap'].unsqueeze(0),
            'line_heatmap': sample['line_heatmap'].unsqueeze(0)
        }
        
        outputs = model(image)
        losses = criterion(outputs, targets)
        
        assert losses['total'].item() > 0
    
    def test_predictor_api_compatibility(self):
        """Test SOLD2Predictor has same API as EdgePredictor."""
        from model.predictor import EdgePredictor
        
        edge_pred = EdgePredictor()
        sold2_pred = SOLD2Predictor(device='cpu')
        
        keypoints = np.array([[50, 50], [100, 100], [150, 50]], dtype=float)
        img_shape = (256, 256)
        
        # Both should have predict method with same signature
        edge_map1 = edge_pred.predict(keypoints, img_shape)
        edge_map2 = sold2_pred.predict(keypoints, img_shape)
        
        assert edge_map1.shape == edge_map2.shape
        assert edge_map1.dtype == edge_map2.dtype


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
