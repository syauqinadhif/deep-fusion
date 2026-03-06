#!/usr/bin/env python3
"""
Training script for DeepFusion on RTX 4080.
Optimized for maximum performance on NVIDIA GPUs.

Usage:
    python train_rtx4080.py --config ../config_rtx4080.yaml --data_path /path/to/kitti
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

# Check for CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ Using CUDA (NVIDIA GPU)")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
else:
    device = torch.device('cpu')
    print("⚠ CUDA not available, using CPU")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DeepFusion, DeepFusionLite
from datasets import KITTIDataset, get_training_transforms, get_val_transforms, collate_fn
from utils import (
    load_config, save_checkpoint, count_parameters,
    seed_everything, EarlyStopping, AverageMeter, LossTracker, get_lr_scheduler
)


def get_device():
    """Get CUDA device if available."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class Trainer:
    """Trainer for RTX 4080 with CUDA optimizations."""

    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.device = get_device()
        self.setup_directories()

        # Set random seeds
        seed_everything(42)

        # Build model
        print("\n" + "="*60)
        print("Building Model")
        print("="*60)
        self.model = self.build_model()
        self.model.to(self.device)

        # Print model info
        total_params = count_parameters(self.model)
        print(f"Total parameters: {total_params:,}")
        print(f"Device: {self.device}")

        # Enable cuDNN auto-tuner
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print("✓ cuDNN auto-tuner enabled")

        # Build datasets
        print("\n" + "="*60)
        print("Loading Datasets")
        print("="*60)
        self.train_dataset, self.val_dataset = self.build_datasets()

        # Build data loaders
        self.train_loader, self.val_loader = self.build_data_loaders()

        # Build optimizer
        self.optimizer = self.build_optimizer()

        # Build scheduler
        self.scheduler = self.build_scheduler()

        # Mixed precision training (AMP)
        system_config = config.get('system', {})
        self.use_amp = system_config.get('use_amp', True)
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            print("✓ Mixed precision (FP16) enabled")
        else:
            self.scaler = None

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

    def build_model(self):
        """Build model from config."""
        model_config = self.config.get('model', {})

        # Use standard DeepFusion (RTX 4080 can handle it!)
        model = DeepFusion(
            lidar_channels=model_config.get('image_features', 256),
            image_channels=model_config.get('image_features', 256),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_heads=model_config.get('n_heads', 8),
            num_layers=model_config.get('n_layers', 1),
            num_classes=model_config.get('num_classes', 3),
            max_objects=model_config.get('max_objects_per_image', 512),
            image_backbone=model_config.get('image_backbone', 'resnet34'),
            pretrained_image=model_config.get('pretrained', True),
            max_points_per_pillar=model_config.get('max_points_per_pillar', 100),
            max_pillars=model_config.get('max_pillars', 12000),
            voxel_size=model_config.get('voxel_size', [0.16, 0.16, 4.0]),
            point_range=model_config.get('point_range', [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0])
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
            max_objects=self.config['model']['max_objects_per_image'],
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
            max_objects=self.config['model']['max_objects_per_image'],
            voxel_size=self.config['model']['voxel_size'],
            point_range=self.config['model']['point_range']
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        return train_dataset, val_dataset

    def build_data_loaders(self):
        """Build data loaders with RTX 4080 optimizations."""
        train_config = self.config.get('training', {})
        system_config = self.config.get('system', {})

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config.get('batch_size', 8),
            shuffle=True,
            num_workers=system_config.get('num_workers', 12),
            pin_memory=system_config.get('pin_memory', True),
            drop_last=True,
            collate_fn=collate_fn,
            prefetch_factor=2,  # RTX 4080 can handle this
            persistent_workers=True  # Keep workers alive
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_config.get('batch_size', 8),
            shuffle=False,
            num_workers=system_config.get('num_workers', 12),
            pin_memory=system_config.get('pin_memory', True),
            collate_fn=collate_fn,
            persistent_workers=True
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

    def build_scheduler(self):
        """Build learning rate scheduler."""
        train_config = self.config.get('training', {})

        scheduler = get_lr_scheduler(
            self.optimizer,
            scheduler_type=train_config.get('scheduler', 'cosine'),
            epochs=train_config.get('epochs', 80),
            min_lr=train_config.get('min_lr', 0.00001)
        )

        return scheduler

    def train_epoch(self, epoch: int, epochs: int) -> dict:
        """Train for one epoch with mixed precision."""
        self.model.train()
        self.loss_tracker.reset()
        epoch_loss = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            points = batch['points'].to(self.device, non_blocking=True)
            images = batch['images'].to(self.device, non_blocking=True)

            # Build targets
            targets = {
                'heatmap': batch['targets']['heatmap'].to(self.device, non_blocking=True),
                'offset': batch['targets']['offset'].to(self.device, non_blocking=True),
                'size': batch['targets']['size'].to(self.device, non_blocking=True),
                'rotation': batch['targets']['rotation'].to(self.device, non_blocking=True),
                'z_center': batch['targets']['z_center'].to(self.device, non_blocking=True)
            }

            # Forward pass with mixed precision
            aug_params = batch['aug_params'][0]

            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(points, images, aug_params)
                    loss, loss_dict = self.model.compute_loss(output, targets)

                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(points, images, aug_params)
                loss, loss_dict = self.model.compute_loss(output, targets)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

            # Update metrics
            epoch_loss.update(loss.item(), points.size(0))
            self.loss_tracker.update(loss_dict)

            # Update progress bar
            if batch_idx % self.print_interval == 0:
                # GPU memory tracking
                if self.device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'mem': f'{mem_allocated:.1f}GB'
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
                points = batch['points'].to(self.device, non_blocking=True)
                images = batch['images'].to(self.device, non_blocking=True)

                targets = {
                    'heatmap': batch['targets']['heatmap'].to(self.device, non_blocking=True),
                    'offset': batch['targets']['offset'].to(self.device, non_blocking=True),
                    'size': batch['targets']['size'].to(self.device, non_blocking=True),
                    'rotation': batch['targets']['rotation'].to(self.device, non_blocking=True),
                    'z_center': batch['targets']['z_center'].to(self.device, non_blocking=True)
                }

                # Mixed precision for validation too
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(points, images, aug_params=None)
                        loss, loss_dict = self.model.compute_loss(output, targets)
                else:
                    output = self.model(points, images, aug_params=None)
                    loss, loss_dict = self.model.compute_loss(output, targets)

                val_loss.update(loss.item(), points.size(0))
                self.loss_tracker.update(loss_dict)

        return self.loss_tracker.get_average()

    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Training on RTX 4080")
        print("="*60)

        epochs = self.config['training']['epochs']
        save_freq = self.config.get('logging', {}).get('save_freq', 5)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(epoch, epochs)

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total'])
                else:
                    self.scheduler.step()

            # Print epoch summary
            elapsed = time.time() - start_time
            epoch_time = elapsed / epoch
            eta = epoch_time * (epochs - epoch + 1)

            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"  Train Loss: {train_metrics.get('total', 0):.4f}")
            print(f"  Val Loss:   {val_metrics.get('total', 0):.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Time:       {epoch_time/60:.1f} min/epoch")
            print(f"  ETA:        {eta/3600:.1f} hours")

            # Save checkpoint
            is_best = val_metrics.get('total', float('inf')) < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics.get('total', self.best_val_loss)

            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # Early stopping check
            if self.early_stopping(val_metrics.get('total', float('inf'))):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print("="*60)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth.tar')

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth.tar')
            print(f"  ✓ Saved best checkpoint (val_loss: {self.best_val_loss:.4f})")

        # Save periodic
        torch.save(checkpoint, self.checkpoint_dir / f'epoch_{epoch}.pth.tar')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DeepFusion on RTX 4080')

    parser.add_argument(
        '--config',
        type=str,
        default='./config_rtx4080.yaml',
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
