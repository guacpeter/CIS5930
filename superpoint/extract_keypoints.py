"""
Keypoint extraction script using SuperPoint for BIPED dataset.

This script extracts keypoints from RGB images using the SuperPoint algorithm
and saves them for use in edge prediction.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add SuperPoint repo to path
sys.path.insert(0, str(Path(__file__).parent / 'superpoint_repo'))

try:
    import torch
    import cv2
    from superpoint_pytorch import SuperPoint
except ImportError as e:
    print(f"Error: Could not import SuperPoint dependencies: {e}")
    print("Run: pip install torch torchvision opencv-python numpy")
    sys.exit(1)


class KeypointExtractor:
    """Wrapper class for SuperPoint keypoint extraction."""

    def __init__(self, weights_path=None, nms_dist=4, conf_thresh=0.015,
                 nn_thresh=0.7, cuda=True):
        """
        Initialize SuperPoint keypoint extractor.

        Args:
            weights_path: Path to pretrained weights (default: use repo weights)
            nms_dist: Non-maximum suppression distance
            conf_thresh: Confidence threshold for keypoints
            nn_thresh: Nearest neighbor threshold for matching
            cuda: Whether to use CUDA if available
        """
        if weights_path is None:
            weights_path = str(Path(__file__).parent / 'superpoint_repo' /
                             'weights' / 'superpoint_v6_from_tf.pth')

        self.device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
        self.conf_thresh = conf_thresh
        self.nms_dist = nms_dist

        # Initialize model
        self.model = SuperPoint(
            nms_radius=nms_dist,
            detection_threshold=conf_thresh
        ).to(self.device)

        # Load weights
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def extract(self, image_path):
        """
        Extract keypoints from an image.

        Args:
            image_path: Path to input image

        Returns:
            dict containing:
                - keypoints: Nx2 array of (x, y) coordinates
                - scores: N array of confidence scores
                - descriptors: 256xN array of descriptors
        """
        # Read image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert to tensor
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # Add batch and channel dims

        # Extract keypoints
        with torch.no_grad():
            result = self.model({'image': img_tensor})

        # Get results (batch size 1)
        keypoints = result['keypoints'][0].cpu().numpy()
        scores = result['keypoint_scores'][0].cpu().numpy()
        descriptors = result['descriptors'][0].cpu().numpy().T  # 256xN

        return {
            'keypoints': keypoints,  # Nx2
            'scores': scores,
            'descriptors': descriptors
        }
    
    def extract_batch(self, image_dir, output_dir, pattern='*.jpg'):
        """
        Extract keypoints from all images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save keypoint files
            pattern: File pattern to match (default: *.jpg)
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(image_dir.glob(pattern))
        print(f"Found {len(image_files)} images in {image_dir}")
        
        for i, img_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
            
            try:
                result = self.extract(img_path)
                
                # Save keypoints
                output_path = output_dir / f"{img_path.stem}_keypoints.npz"
                np.savez(
                    output_path,
                    keypoints=result['keypoints'],
                    scores=result['scores'],
                    descriptors=result['descriptors']
                )
                
                print(f"  Extracted {len(result['keypoints'])} keypoints")
                
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
        
        print(f"\nKeypoint extraction complete. Saved to {output_dir}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Extract keypoints from images using SuperPoint'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input image file or directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for keypoint files'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.jpg',
        help='File pattern for batch processing (default: *.jpg)'
    )
    parser.add_argument(
        '--conf-thresh',
        type=float,
        default=0.015,
        help='Confidence threshold (default: 0.015)'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = KeypointExtractor(
        conf_thresh=args.conf_thresh,
        cuda=not args.no_cuda
    )
    
    # Process
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        result = extractor.extract(input_path)
        output_path = Path(args.output) / f"{input_path.stem}_keypoints.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            keypoints=result['keypoints'],
            scores=result['scores'],
            descriptors=result['descriptors']
        )
        print(f"Extracted {len(result['keypoints'])} keypoints")
        print(f"Saved to {output_path}")
    else:
        # Directory
        extractor.extract_batch(input_path, args.output, args.pattern)


if __name__ == '__main__':
    main()
