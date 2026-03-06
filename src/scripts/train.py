#!/usr/bin/env python3
"""
Training script for DeepFusion 3D Object Detection.

Usage:
    python train.py --config ../config.yaml --data_path /path/to/kitti
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DeepFusion, DeepFusionLite, build_deepfusion_model
from datasets import KITTIDataset, get_training_transforms, get_val_transforms, collate_fn
from utils import (
    load_config, save_checkpoint, count_parameters, get_device,
    seed_everything, EarlyStopping, get_lr_scheduler, AverageMeter, LossTracker
)


class Trainer:
    """
    Trainer class for DeepFusion model.
    """

    def __init__(self, config: dict, args):
        self.config = config
        self.args = args

        # Setup
        self.device = get_device()
        self.setup_directories()

        # Set random seeds
        if 'seed' in config.get('training', {}):
            seed_everything(config['training']['seed'])

        # Build model
        self.model = self.build_model()
        self.model.to(self.device)

        # Print model info
        print(f"\n{'='*60}")
        print("Model Architecture")
        print(f"{'='*60}")
        model_info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        for key, value in model_info.items():
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
        print(f"  Total parameters: {count_parameters(self.model):,}")
        print(f"{'='*60}\n")

        # Build datasets
        self.train_dataset, self.val_dataset = self.build_datasets()

        # Build data loaders
        self.train_loader, self.val_loader = self.build_data_loaders()

        # Build optimizer and scheduler
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # Early stopping
        early_stop_config = config.get('training', {}).get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 15),
            min_delta=early_stop_config.get('min_delta', 0.001),
            mode='min'
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.loss_tracker = LossTracker()

        # Logging
        self.log_interval = config.get('logging', {}).get('log_freq', 100)
        self.print_interval = config.get('logging', {}).get('print_freq', 10)

    def setup_directories(self):
        """Setup output directories."""
        self.output_dir = Path(self.args.output_dir) if self.args.output_dir else Path('./results')
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        save_config(self.config, self.output_dir / 'config.yaml')

    def build_model(self) -> nn.Module:
        """Build model from config."""
        model_type = self.args.model_type if self.args.model_type else 'standard'

        if model_type == 'lite':
            print("Building DeepFusion Lite model...")
            model = DeepFusionLite(
                num_classes=self.config['model']['num_classes'],
                max_objects=self.config['model']['max_objects_per_image'],
                max_points_per_pillar=self.config['model']['max_points_per_pillar'],
                max_pillars=self.config['model']['max_pillars'],
                voxel_size=self.config['model']['voxel_size'],
                point_range=self.config['model']['point_range']
            )
        else:
            print("Building DeepFusion model...")
            model = build_deepfusion_model(self.config)

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
            max_objects=self.config['model']['max_objects_per_image']
        )

        # Validation dataset
        val_transforms = get_val_transforms(self.config)
        val_dataset = KITTIDataset(
            root_path=self.args.data_path,
            split='val',
            transform=val_transforms,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            max_objects=self.config['model']['max_objects_per_image']
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        return train_dataset, val_dataset

    def build_data_loaders(self):
        """Build data loaders."""
        train_config = self.config.get('training', {})
        system_config = self.config.get('system', {})

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config.get('batch_size', 4),
            shuffle=True,
            num_workers=system_config.get('num_workers', 4),
            pin_memory=system_config.get('pin_memory', True),
            drop_last=True,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_config.get('batch_size', 4),
            shuffle=False,
            num_workers=system_config.get('num_workers', 4),
            pin_memory=system_config.get('pin_memory', True),
            collate_fn=collate_fn
        )

        return train_loader, val_loader

    def build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config."""
        train_config = self.config.get('training', {})

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_config.get('learning_rate', 0.001),
            weight_decay=train_config.get('weight_decay', 0.0001)
        )

        return optimizer

    def build_scheduler(self):
        """Build learning rate scheduler."""
        train_config = self.config.get('training', {})

        scheduler = get_lr_scheduler(
            self.optimizer,
            scheduler_type=train_config.get('scheduler', 'cosine'),
            epochs=train_config.get('epochs', 100),
            min_lr=train_config.get('min_lr', 0.00001)
        )

        return scheduler

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        self.loss_tracker.reset()
        epoch_loss = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['epochs']}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            points = batch['points'].to(self.device)
            images = batch['images'].to(self.device)

            # Build targets (placeholder - implement proper target generation)
            targets = {
                'heatmap': batch['targets']['heatmap'].to(self.device),
                'offset': batch['targets']['offset'].to(self.device),
                'size': batch['targets']['size'].to(self.device),
                'rotation': batch['targets']['rotation'].to(self.device),
                'z_center': batch['targets']['z_center'].to(self.device)
            }

            # Forward pass
            aug_params = batch['aug_params'][0]  # Use first sample's aug params
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
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss.avg:.4f}'
                })

        # Compute epoch averages
        avg_losses = self.loss_tracker.get_average()

        return {
            'total_loss': epoch_loss.avg,
            **avg_losses
        }

    def validate(self) -> dict:
        """Validate the model."""
        self.model.eval()

        val_loss = AverageMeter()
        self.loss_tracker.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                points = batch['points'].to(self.device)
                images = batch['images'].to(self.device)

                targets = {
                    'heatmap': batch['targets']['heatmap'].to(self.device),
                    'offset': batch['targets']['offset'].to(self.device),
                    'size': batch['targets']['size'].to(self.device),
                    'rotation': batch['targets']['rotation'].to(self.device),
                    'z_center': batch['targets']['z_center'].to(self.device)
                }

                # Forward pass
                output = self.model(points, images, aug_params=None)

                # Compute loss
                loss, loss_dict = self.model.compute_loss(output, targets)

                # Update metrics
                val_loss.update(loss.item(), points.size(0))
                self.loss_tracker.update(loss_dict)

        # Compute averages
        avg_losses = self.loss_tracker.get_average()

        return {
            'total_loss': val_loss.avg,
            **avg_losses
        }

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}\n")

        epochs = self.config['training']['epochs']
        save_freq = self.config.get('logging', {}).get('save_freq', 5)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()

            # Print epoch summary
            print(f"\nEpoch {epoch}/{epochs} Summary:")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['total_loss']:.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']

            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # Early stopping check
            if self.early_stopping(val_metrics['total_loss']):
                print(f"\nEarly stopping triggered after epoch {epoch}")
                break

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save latest checkpoint
        save_path = self.checkpoint_dir / 'latest.pth.tar'
        save_checkpoint(checkpoint, self.checkpoint_dir, 'latest.pth.tar')

        # Save best checkpoint
        if is_best:
            save_path = self.checkpoint_dir / 'best.pth.tar'
            torch.save(checkpoint, save_path)
            print(f"  Saved best checkpoint: {save_path}")

        # Save periodic checkpoint
        save_path = self.checkpoint_dir / f'epoch_{epoch}.pth.tar'
        torch.save(checkpoint, save_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DeepFusion model')

    parser.add_argument(
        '--config',
        type=str,
        default='./config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to KITTI dataset root directory'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        choices=['standard', 'lite'],
        default='standard',
        help='Model type: standard or lite'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override seed if specified
    if args.seed is not None:
        config['training']['seed'] = args.seed

    # Create trainer
    trainer = Trainer(config, args)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['best_val_loss']

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
