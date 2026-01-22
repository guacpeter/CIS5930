# SuperPoint Keypoint Extraction

This directory contains the SuperPoint algorithm for extracting keypoints from RGB images.

## Contents

- `superpoint_repo/` - Cloned SuperPoint repository from https://github.com/rpautrat/SuperPoint
- `extract_keypoints.py` - Wrapper script for easy keypoint extraction from BIPED dataset

## Setup

1. Install dependencies:
```bash
pip install -r superpoint_repo/requirements.txt
```

Required packages:
- torch
- opencv-python
- numpy

## Usage

### Extract keypoints from all BIPED images:

```bash
python superpoint/extract_keypoints.py \
    --input data/rgb \
    --output data/keypoints \
    --pattern "*.jpg"
```

### Extract from a single image:

```bash
python superpoint/extract_keypoints.py \
    --input data/rgb/RGB_001.jpg \
    --output data/keypoints
```

### Options:

- `--input, -i`: Input image file or directory (required)
- `--output, -o`: Output directory for keypoint files (required)
- `--pattern`: File pattern for batch processing (default: `*.jpg`)
- `--conf-thresh`: Confidence threshold for keypoints (default: 0.015)
- `--no-cuda`: Disable CUDA even if available

## Output Format

Keypoints are saved as `.npz` files containing:
- `keypoints`: Nx2 array of (x, y) coordinates
- `scores`: N array of confidence scores  
- `descriptors`: 256xN array of feature descriptors

## Example

```python
from superpoint.extract_keypoints import KeypointExtractor
import numpy as np

# Initialize extractor
extractor = KeypointExtractor()

# Extract keypoints
result = extractor.extract('data/rgb/RGB_001.jpg')

print(f"Found {len(result['keypoints'])} keypoints")
print(f"Keypoint coordinates shape: {result['keypoints'].shape}")
print(f"Descriptor shape: {result['descriptors'].shape}")

# Load saved keypoints
data = np.load('data/keypoints/RGB_001_keypoints.npz')
keypoints = data['keypoints']
scores = data['scores']
descriptors = data['descriptors']
```

## Reference

SuperPoint: Self-Supervised Interest Point Detection and Description
- Paper: https://arxiv.org/abs/1712.07629
- Repository: https://github.com/rpautrat/SuperPoint
