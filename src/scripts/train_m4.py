#!/usr/bin/env python3
"""
Training script for DeepFusion on Apple Silicon (M4 Pro).

Usage:
    python train_m4.py --config ../config_m4_pro.yaml --data_path ~/Datasets/KITTI
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Apple Silicon MPS support
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"✓ Using Apple Silicon GPU (MPS)")
    print(f"  MPS memory: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB allocated")
else:
    device = torch.device('cpu')
    print("⚠ MPS not available, using CPU")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DeepFusion, DeepFusionLite
from datasets import KITTIDataset, get_training_transforms, get_val_transforms, collate_fn
from utils import (
    load_config, save_checkpoint, count_parameters,
    seed_everything, EarlyStopping, AverageMeter, LossTracker
)


def get_device():
    """Get the best available device for M4 Pro."""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ Apple Silicon GPU detected")
        print(f"  Allocated memory: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")
        return device
    return torch.device('cpu')


class M4Trainer:
    """Trainer optimized for M4 Pro."""

    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.device = get_device()
        self.setup_directories()

        # Set random seeds
        seed_everything(42)

        # Build model
        print("\n" + "="*60)
        print("Building Model for M4 Pro Training")
        print("="*60)
        self.model = self.build_model()
        self.model.to(self.device)

        # Print model info
        total_params = count_parameters(self.model)
        print(f"Total parameters: {total_params:,}")
        print(f"Device: {self.device}")

        # Build datasets
        print("\n" + "="*60)
        print("Loading Datasets")
        print("="*60)
        self.train_dataset, self.val_dataset = self.build_datasets()

        # Build data loaders (optimized for M4 Pro)
        self.train_loader, self.val_loader = self.build_data_loaders()

        # Build optimizer
        self.optimizer = self.build_optimizer()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.loss_tracker = LossTracker()

        # Logging
        self.log_interval = 50
        self.print_interval = 10

    def setup_directories(self):
        """Setup output directories."""
        self.output_dir = Path(self.args.output_dir) if self.args.output_dir else Path('./results')
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def build_model(self):
        """Build model optimized for M4 Pro."""
        # Use lite model by default for faster training
        # But ensure consistent channel dimensions
        model = DeepFusionLite(
            lidar_channels=256,      # Must match PointPillars output
            image_channels=256,      # Must match ResNet output
            hidden_dim=256,          # Hidden dimension for alignment
            num_heads=8,             # Number of attention heads
            num_classes=3,
            max_objects=512,
            max_points_per_pillar=100,
            max_pillars=12000,
            voxel_size=[0.16, 0.16, 4.0],
            point_range=[-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
        )
        return model

    def build_datasets(self):
        """Build training and validation datasets."""
        data_config = self.config.get('data', {})

        # Training dataset
        train_transforms = get_training_transforms(self.config)
        train_dataset = KITTIDataset(
            root_path=self.args.data_path,
            split='train',
            transform=train_transforms,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            max_objects=512,
            voxel_size=self.config['model']['voxel_size'],
            point_range=self.config['model']['point_range']
        )

        # Validation dataset
        val_transforms = get_val_transforms(self.config)
        val_dataset = KITTIDataset(
            root_path=self.args.data_path,
            split='val',
            transform=val_transforms,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            max_objects=512,
            voxel_size=self.config['model']['voxel_size'],
            point_range=self.config['model']['point_range']
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        return train_dataset, val_dataset

    def build_data_loaders(self):
        """Build data loaders optimized for M4 Pro."""
        train_config = self.config.get('training', {})

        # M4 Pro optimization: fewer workers to avoid memory issues
        num_workers = min(4, os.cpu_count())

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config.get('batch_size', 2),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,  # Not needed for MPS
            drop_last=True,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,  # Use batch size 1 for validation
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            collate_fn=collate_fn
        )

        return train_loader, val_loader

    def build_optimizer(self):
        """Build optimizer."""
        train_config = self.config.get('training', {})

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_config.get('learning_rate', 0.001),
            weight_decay=train_config.get('weight_decay', 0.0001)
        )

        return optimizer

    def train_epoch(self, epoch: int, epochs: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        self.loss_tracker.reset()
        epoch_loss = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            points = batch['points'].to(self.device)
            images = batch['images'].to(self.device)

            # Build targets
            targets = {
                'heatmap': batch['targets']['heatmap'].to(self.device),
                'offset': batch['targets']['offset'].to(self.device),
                'size': batch['targets']['size'].to(self.device),
                'rotation': batch['targets']['rotation'].to(self.device),
                'z_center': batch['targets']['z_center'].to(self.device)
            }

            # Forward pass
            aug_params = batch['aug_params'][0]
            output = self.model(points, images, aug_params)

            # Compute loss
            loss, loss_dict = self.model.compute_loss(output, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            # Optimizer step
            self.optimizer.step()

            # Update metrics
            epoch_loss.update(loss.item(), points.size(0))
            self.loss_tracker.update(loss_dict)

            # Update progress bar
            if batch_idx % self.print_interval == 0:
                # MPS memory tracking
                if self.device.type == 'mps':
                    mem_gb = torch.mps.current_allocated_memory() / 1024**3
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'mem': f'{mem_gb:.1f}GB'
                    })
                else:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return self.loss_tracker.get_average()

    def validate(self) -> dict:
        """Validate the model."""
        self.model.eval()
        val_loss = AverageMeter()
        self.loss_tracker.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                points = batch['points'].to(self.device)
                images = batch['images'].to(self.device)

                targets = {
                    'heatmap': batch['targets']['heatmap'].to(self.device),
                    'offset': batch['targets']['offset'].to(self.device),
                    'size': batch['targets']['size'].to(self.device),
                    'rotation': batch['targets']['rotation'].to(self.device),
                    'z_center': batch['targets']['z_center'].to(self.device)
                }

                output = self.model(points, images, aug_params=None)
                loss, loss_dict = self.model.compute_loss(output, targets)

                val_loss.update(loss.item(), points.size(0))
                self.loss_tracker.update(loss_dict)

        return self.loss_tracker.get_average()

    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Training on M4 Pro")
        print("="*60)

        epochs = self.config['training']['epochs']
        save_freq = self.config.get('logging', {}).get('save_freq', 10)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(epoch, epochs)

            # Validate
            val_metrics = self.validate()

            # Print summary
            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"  Train Loss: {train_metrics.get('total', 0):.4f}")
            print(f"  Val Loss:   {val_metrics.get('total', 0):.4f}")

            # Save checkpoint
            is_best = val_metrics.get('total', float('inf')) < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics.get('total', self.best_val_loss)

            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print("="*60)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth.tar')

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth.tar')
            print(f"  ✓ Saved best checkpoint (val_loss: {self.best_val_loss:.4f})")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DeepFusion on M4 Pro')

    parser.add_argument(
        '--config',
        type=str,
        default='./config_m4_pro.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to KITTI dataset'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Create trainer
    trainer = M4Trainer(config, args)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
