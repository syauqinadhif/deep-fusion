"""
Common utility functions for DeepFusion training and evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
import os


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    state: Dict,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth.tar'
) -> None:
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """Load model checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path)

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Get learning rate scheduler."""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('epochs', 100),
            eta_min=kwargs.get('min_lr', 1e-5)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10)
        )
    else:
        return None


def convert_box_format(boxes: torch.Tensor, from_format: str, to_format: str) -> torch.Tensor:
    """
    Convert bounding box between different formats.

    Formats:
    - 'xyzwlh': [x, y, z, w, l, h] (center-based)
    - 'xyzxyzz': [x1, y1, z1, x2, y2, z2] (corner-based)
    - ' kitti': [x, y, z, l, w, h, yaw] (KITTI format)

    Args:
        boxes: (N, 6) or (N, 7) tensor
        from_format: Source format
        to_format: Target format

    Returns:
        Converted boxes tensor
    """
    if from_format == to_format:
        return boxes

    # Convert from center-based to corner-based
    if from_format == 'xyzwlh' and to_format == 'xyzxyzz':
        x, y, z, w, l, h = boxes.unbind(dim=-1)
        return torch.stack([
            x - w/2, y - l/2, z - h/2,
            x + w/2, y + l/2, z + h/2
        ], dim=-1)

    # Convert from corner-based to center-based
    if from_format == 'xyzxyzz' and to_format == 'xyzwlh':
        x1, y1, z1, x2, y2, z2 = boxes.unbind(dim=-1)
        return torch.stack([
            (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2,
            x2 - x1, y2 - y1, z2 - z1
        ], dim=-1)

    raise ValueError(f"Unsupported conversion: {from_format} -> {to_format}")
