#!/usr/bin/env python3
"""
SOLD2 Training Script - Unsupervised Line Detection.

Three-phase training pipeline:
1. Synthetic pre-training: Train on generated geometric shapes
2. Homography adaptation: Generate pseudo ground truth on real images
3. Real image fine-tuning: Train on BIPED with pseudo labels

Usage:
    python scripts/train_sold2.py --phase synthetic --epochs 50
    python scripts/train_sold2.py --phase adapt --checkpoint sold2_synth.pt
    python scripts/train_sold2.py --phase finetune --checkpoint sold2_synth.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm

from model.sold2_network import SOLD2Net, create_sold2_model, KeypointConditionedSOLD2
from model.sold2_training import (
    SyntheticLineDataset,
    BIPEDKeypointDataset,
    HomographyAdaptation,
    SOLD2Loss,
    PhotometricAugmentation
)


def custom_collate_fn(batch):
    """Custom collate function that handles variable-length fields."""
    images = torch.stack([item['image'] for item in batch])
    junction_heatmaps = torch.stack([item['junction_heatmap'] for item in batch])
    line_heatmaps = torch.stack([item['line_heatmap'] for item in batch])
    # Don't collate 'lines' - keep as list
    lines = [item['lines'] for item in batch]
    
    return {
        'image': images,
        'junction_heatmap': junction_heatmaps,
        'line_heatmap': line_heatmaps,
        'lines': lines
    }


def keypoint_collate_fn(batch):
    """Collate function for keypoint-conditioned training."""
    images = torch.stack([item['image'] for item in batch])
    keypoint_maps = torch.stack([item['keypoint_map'] for item in batch])
    junction_heatmaps = torch.stack([item['junction_heatmap'] for item in batch])
    line_heatmaps = torch.stack([item['line_heatmap'] for item in batch])
    # Keep keypoints as list since they have variable length
    keypoints = [item['keypoints'] for item in batch]
    scores = [item['scores'] for item in batch]
    
    return {
        'image': images,
        'keypoint_map': keypoint_maps,
        'junction_heatmap': junction_heatmaps,
        'line_heatmap': line_heatmaps,
        'keypoints': keypoints,
        'scores': scores
    }


def train_synthetic(args):
    """Phase 1: Train on synthetic dataset."""
    print("=" * 60)
    print("Phase 1: Synthetic Pre-training")
    print("=" * 60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset - use smaller images for faster training if lightweight
    img_size = (128, 128) if args.lightweight else (256, 256)
    train_dataset = SyntheticLineDataset(
        num_samples=args.num_samples,
        img_size=img_size,
        min_lines=3,
        max_lines=15
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Use custom collate function
    )
    
    # Create model - use smaller base_channels for lightweight mode
    base_channels = 16 if args.lightweight else 64
    descriptor_dim = 32 if args.lightweight else 128
    model = SOLD2Net(in_channels=1, base_channels=base_channels, descriptor_dim=descriptor_dim)
    model = model.to(device)
    print(f"Model: base_channels={base_channels}, descriptor_dim={descriptor_dim}")
    
    # Loss and optimizer
    criterion = SOLD2Loss(junction_weight=1.0, line_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            targets = {
                'junction_heatmap': batch['junction_heatmap'].to(device),
                'line_heatmap': batch['line_heatmap'].to(device)
            }
            
            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            
            # Loss
            losses = criterion(outputs, targets)
            loss = losses['total']
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(args.output_dir) / 'sold2_synth_best.pt'
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")
    
    # Save final model
    final_path = Path(args.output_dir) / 'sold2_synth_final.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    
    return model


def train_keypoint_conditioned(args):
    """Train keypoint-conditioned SOLD2 on BIPED with SuperPoint keypoints."""
    print("=" * 60)
    print("Keypoint-Conditioned SOLD2 Training")
    print("=" * 60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset
    img_size = (256, 256) if not args.lightweight else (128, 128)
    train_dataset = BIPEDKeypointDataset(
        data_dir=args.data_dir,
        img_size=img_size,
        keypoint_sigma=4.0,
        augment=True
    )
    
    if len(train_dataset) == 0:
        print("Error: No samples found. Make sure data/rgb/, data/keypoints/, and data/edges/ exist.")
        return None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=keypoint_collate_fn
    )
    
    # Create keypoint-conditioned model
    base_channels = 16 if args.lightweight else 64
    descriptor_dim = 32 if args.lightweight else 128
    model = KeypointConditionedSOLD2(
        base_channels=base_channels,
        descriptor_dim=descriptor_dim,
        keypoint_sigma=4.0
    )
    
    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        # Load with strict=False since architecture differs slightly
        model.load_state_dict(state_dict, strict=False)
        print("  Warning: Loaded with strict=False (architecture differs)")
    
    model = model.to(device)
    print(f"Model: KeypointConditionedSOLD2 (base_channels={base_channels}, descriptor_dim={descriptor_dim})")
    
    # Loss and optimizer
    criterion = SOLD2Loss(junction_weight=1.0, line_weight=2.0)  # Higher weight for lines
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_line_loss = 0.0
        epoch_junc_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            keypoint_maps = batch['keypoint_map'].to(device)
            targets = {
                'junction_heatmap': batch['junction_heatmap'].to(device),
                'line_heatmap': batch['line_heatmap'].to(device)
            }
            
            # Forward with keypoint conditioning
            optimizer.zero_grad()
            outputs = model(images, keypoint_map=keypoint_maps)
            
            # Compute loss
            losses = criterion(outputs, targets)
            loss = losses['total']
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if 'line' in losses:
                epoch_line_loss += losses['line'].item()
            if 'junction' in losses:
                epoch_junc_loss += losses['junction'].item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'line': f"{losses.get('line', 0):.4f}"
            })
        
        scheduler.step()
        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_line = epoch_line_loss / n_batches
        avg_junc = epoch_junc_loss / n_batches
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} (Line={avg_line:.4f}, Junc={avg_junc:.4f})")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(args.output_dir) / 'kp_sold2_best.pt'
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = Path(args.output_dir) / f'kp_sold2_epoch{epoch+1}.pt'
            torch.save(model.state_dict(), ckpt_path)
    
    # Save final model
    final_path = Path(args.output_dir) / 'kp_sold2_final.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    
    return model


def adapt_homography(args):
    """Phase 2: Generate pseudo ground truth using homography adaptation."""
    print("=" * 60)
    print("Phase 2: Homography Adaptation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load pre-trained model with matching architecture
    base_channels = 16 if args.lightweight else 64
    descriptor_dim = 32 if args.lightweight else 128
    model = SOLD2Net(in_channels=1, base_channels=base_channels, descriptor_dim=descriptor_dim)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"Loaded model: base_channels={base_channels}, descriptor_dim={descriptor_dim}")
    
    # Homography adapter
    adapter = HomographyAdaptation(
        num_homographies=args.num_homographies,
        aggregation_method='mean',
        min_confidence=0.3
    )
    
    # Load real images
    data_dir = Path(args.data_dir)
    rgb_dir = data_dir / 'rgb'
    output_dir = Path(args.output_dir) / 'pseudo_gt'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = sorted(rgb_dir.glob('*.jpg'))
    print(f"Processing {len(image_paths)} images...")
    
    for img_path in tqdm(image_paths, desc="Generating pseudo GT"):
        # Load and preprocess image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Resize for network
        H, W = img.shape
        new_H = ((H + 31) // 32) * 32
        new_W = ((W + 31) // 32) * 32
        img_resized = cv2.resize(img, (new_W, new_H))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)
        
        # Adapt
        pseudo_gt = adapter.adapt(model, img_tensor, device)
        
        # Save pseudo ground truth
        stem = img_path.stem
        junc_path = output_dir / f"{stem}_junction.npy"
        line_path = output_dir / f"{stem}_line.npy"
        
        np.save(junc_path, pseudo_gt['junction_heatmap'].cpu().numpy())
        np.save(line_path, pseudo_gt['line_heatmap'].cpu().numpy())
    
    print(f"\nPseudo GT saved to {output_dir}")


def train_finetune(args):
    """Phase 3: Fine-tune on real images with pseudo ground truth."""
    print("=" * 60)
    print("Phase 3: Fine-tuning on Real Images")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load pre-trained model with matching architecture
    base_channels = 16 if args.lightweight else 64
    descriptor_dim = 32 if args.lightweight else 128
    model = SOLD2Net(in_channels=1, base_channels=base_channels, descriptor_dim=descriptor_dim)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    model = model.to(device)
    print(f"Model: base_channels={base_channels}, descriptor_dim={descriptor_dim}")
    
    # Create dataset
    data_dir = Path(args.data_dir)
    pseudo_dir = Path(args.output_dir) / 'pseudo_gt'
    
    dataset = RealImageDataset(data_dir, pseudo_dir)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Loss and optimizer
    criterion = SOLD2Loss(junction_weight=1.0, line_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)  # Lower LR for fine-tuning
    augment = PhotometricAugmentation()
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            
            # Apply augmentation
            images = augment(images)
            
            targets = {
                'junction_heatmap': batch['junction_heatmap'].to(device),
                'line_heatmap': batch['line_heatmap'].to(device)
            }
            
            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            
            # Loss
            losses = criterion(outputs, targets)
            loss = losses['total']
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(args.output_dir) / 'sold2_finetuned_best.pt'
            torch.save(model.state_dict(), save_path)
    
    # Save final model
    final_path = Path(args.output_dir) / 'sold2_finetuned_final.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\nFine-tuning complete. Final model saved to {final_path}")


class RealImageDataset(torch.utils.data.Dataset):
    """Dataset for real images with pseudo ground truth."""
    
    def __init__(self, data_dir, pseudo_dir):
        self.data_dir = Path(data_dir)
        self.pseudo_dir = Path(pseudo_dir)
        
        # Find images with corresponding pseudo GT
        rgb_dir = self.data_dir / 'rgb'
        self.samples = []
        
        for img_path in rgb_dir.glob('*.jpg'):
            stem = img_path.stem
            junc_path = self.pseudo_dir / f"{stem}_junction.npy"
            line_path = self.pseudo_dir / f"{stem}_line.npy"
            
            if junc_path.exists() and line_path.exists():
                self.samples.append({
                    'image': img_path,
                    'junction': junc_path,
                    'line': line_path
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = cv2.imread(str(sample['image']), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        new_H = ((H + 31) // 32) * 32
        new_W = ((W + 31) // 32) * 32
        img = cv2.resize(img, (new_W, new_H))
        
        # Load pseudo GT
        junction = np.load(sample['junction'])
        line = np.load(sample['line'])
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        junc_tensor = torch.from_numpy(junction).float().squeeze()
        line_tensor = torch.from_numpy(line).float().squeeze()
        
        # Ensure correct shape
        if junc_tensor.dim() == 4:
            junc_tensor = junc_tensor[0]
        if line_tensor.dim() == 4:
            line_tensor = line_tensor[0]
        
        return {
            'image': img_tensor,
            'junction_heatmap': junc_tensor,
            'line_heatmap': line_tensor
        }


def main():
    parser = argparse.ArgumentParser(description="SOLD2 Training Pipeline")
    
    parser.add_argument('--phase', type=str, required=True,
                        choices=['synthetic', 'adapt', 'finetune', 'keypoint'],
                        help="Training phase: synthetic, adapt, finetune, or keypoint (keypoint-conditioned)")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--num_samples', type=int, default=10000,
                        help="Number of synthetic samples")
    parser.add_argument('--num_homographies', type=int, default=50,
                        help="Number of homographies for adaptation")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to checkpoint for resume/finetune")
    parser.add_argument('--data_dir', type=str, default='data',
                        help="Path to data directory")
    parser.add_argument('--output_dir', type=str, default='results/sold2',
                        help="Output directory for models")
    parser.add_argument('--lightweight', action='store_true',
                        help="Use lightweight model for faster CPU training")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.phase == 'synthetic':
        train_synthetic(args)
    elif args.phase == 'adapt':
        if not args.checkpoint:
            print("Error: --checkpoint required for adapt phase")
            sys.exit(1)
        adapt_homography(args)
    elif args.phase == 'finetune':
        train_finetune(args)
    elif args.phase == 'keypoint':
        train_keypoint_conditioned(args)


if __name__ == '__main__':
    main()

