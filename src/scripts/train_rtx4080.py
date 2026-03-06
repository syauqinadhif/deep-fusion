#!/usr/bin/env python3
"""
Training script for DeepFusion on RTX 4080.
OPTIMIZED VERSION:
  1. torch.compile()          — ~20-30% speedup gratis (PyTorch 2.0+)
  2. prefetch_factor=4        — DataLoader lebih agresif prefetch
  3. GPU util monitor         — tampilkan GPU% setiap epoch
  4. AMP + unscale sebelum clip_grad
  5. aug_params diteruskan sebagai list[B]
  6. save_checkpoint aman untuk torch.compile (_orig_mod)
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ Using CUDA — {torch.cuda.get_device_name(0)}")
    print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  CUDA    : {torch.version.cuda}")
    print(f"  PyTorch : {torch.__version__}")
else:
    device = torch.device('cpu')
    print("⚠ CUDA not available, using CPU")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DeepFusion
from datasets import KITTIDataset, get_training_transforms, get_val_transforms, collate_fn
from utils import (
    load_config, count_parameters, seed_everything,
    EarlyStopping, AverageMeter, LossTracker, get_lr_scheduler
)


def get_gpu_util() -> str:
    """Baca GPU utilization % via nvidia-smi."""
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            encoding='utf-8', timeout=2
        ).strip().split(',')
        util  = out[0].strip()
        mem_u = int(out[1].strip())
        mem_t = int(out[2].strip())
        return f"GPU {util}% | VRAM {mem_u}/{mem_t} MiB"
    except Exception:
        return ""


class Trainer:
    def __init__(self, config: dict, args):
        self.config = config
        self.args   = args
        self.device = device
        self.setup_directories()

        seed_everything(42)

        # ── Model ─────────────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("Building Model")
        print("="*60)
        self.model = self.build_model().to(self.device)
        print(f"Parameters : {count_parameters(self.model):,}")

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print("✓ cuDNN benchmark enabled")

            # torch.compile — ~20-30% speedup gratis di PyTorch >= 2.0
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("✓ torch.compile enabled (mode=reduce-overhead)")
            except Exception as e:
                print(f"⚠ torch.compile skip: {e}")

        # ── Datasets & loaders ────────────────────────────────────────────────
        print("\n" + "="*60)
        print("Loading Datasets")
        print("="*60)
        self.train_dataset, self.val_dataset = self.build_datasets()
        self.train_loader,  self.val_loader  = self.build_data_loaders()

        # ── Optimizer & scheduler ─────────────────────────────────────────────
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # ── Mixed precision ───────────────────────────────────────────────────
        sys_cfg      = config.get('system', {})
        self.use_amp = sys_cfg.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler  = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("✓ Mixed precision (FP16) enabled")

        # ── Early stopping ────────────────────────────────────────────────────
        es_cfg = config.get('training', {}).get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience  = es_cfg.get('patience',  15),
            min_delta = es_cfg.get('min_delta', 0.001),
            mode      = 'min'
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.current_epoch  = 0
        self.best_val_loss  = float('inf')
        self.loss_tracker   = LossTracker()
        self.print_interval = config.get('logging', {}).get('print_freq', 10)

    # ── directories ───────────────────────────────────────────────────────────
    def setup_directories(self):
        self.output_dir     = Path(self.args.output_dir) if self.args.output_dir else Path('./results')
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir        = self.output_dir / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ── builders ──────────────────────────────────────────────────────────────
    def build_model(self):
        mc = self.config.get('model', {})
        return DeepFusion(
            lidar_channels        = mc.get('image_features',       256),
            image_channels        = mc.get('image_features',       256),
            hidden_dim            = mc.get('hidden_dim',           256),
            num_heads             = mc.get('n_heads',                8),
            num_layers            = mc.get('n_layers',               1),
            num_classes           = mc.get('num_classes',            3),
            max_objects           = mc.get('max_objects_per_image', 512),
            image_backbone        = mc.get('image_backbone',  'resnet34'),
            pretrained_image      = mc.get('pretrained',          True),
            max_points_per_pillar = mc.get('max_points_per_pillar', 100),
            max_pillars           = mc.get('max_pillars',        12000),
            voxel_size            = mc.get('voxel_size',  [0.16, 0.16, 4.0]),
            point_range           = mc.get('point_range',
                                           [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0])
        )

    def build_datasets(self):
        train_dataset = KITTIDataset(
            root_path   = self.args.data_path,
            split       = 'train',
            transform   = get_training_transforms(self.config),
            class_names = ['Car', 'Pedestrian', 'Cyclist'],
            max_objects = self.config['model']['max_objects_per_image'],
            voxel_size  = self.config['model']['voxel_size'],
            point_range = self.config['model']['point_range']
        )
        val_dataset = KITTIDataset(
            root_path   = self.args.data_path,
            split       = 'val',
            transform   = get_val_transforms(self.config),
            class_names = ['Car', 'Pedestrian', 'Cyclist'],
            max_objects = self.config['model']['max_objects_per_image'],
            voxel_size  = self.config['model']['voxel_size'],
            point_range = self.config['model']['point_range']
        )
        print(f"Train : {len(train_dataset)} samples")
        print(f"Val   : {len(val_dataset)} samples")
        return train_dataset, val_dataset

    def build_data_loaders(self):
        tc = self.config.get('training', {})
        sc = self.config.get('system',   {})
        bs = tc.get('batch_size',  8)
        nw = sc.get('num_workers', 6)   # 6 lebih stabil dari 12 untuk workload ini

        train_loader = DataLoader(
            self.train_dataset,
            batch_size         = bs,
            shuffle            = True,
            num_workers        = nw,
            pin_memory         = sc.get('pin_memory', True),
            drop_last          = True,
            collate_fn         = collate_fn,
            prefetch_factor    = 4,          # ← naik dari 2, supply data lebih cepat
            persistent_workers = True
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size         = bs,
            shuffle            = False,
            num_workers        = nw,
            pin_memory         = sc.get('pin_memory', True),
            collate_fn         = collate_fn,
            prefetch_factor    = 4,
            persistent_workers = True
        )
        print(f"DataLoader : batch={bs}, workers={nw}, prefetch=4")
        return train_loader, val_loader

    def build_optimizer(self):
        tc = self.config.get('training', {})
        return optim.AdamW(
            self.model.parameters(),
            lr           = tc.get('learning_rate', 1e-3),
            weight_decay = tc.get('weight_decay',  1e-4)
        )

    def build_scheduler(self):
        tc = self.config.get('training', {})
        return get_lr_scheduler(
            self.optimizer,
            scheduler_type = tc.get('scheduler', 'cosine'),
            epochs         = tc.get('epochs',    80),
            min_lr         = tc.get('min_lr',    1e-5)
        )

    # ── training epoch ────────────────────────────────────────────────────────
    def train_epoch(self, epoch: int, epochs: int) -> dict:
        self.model.train()
        self.loss_tracker.reset()

        pbar        = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}")
        batch_times = []
        t0          = time.time()

        for batch_idx, batch in enumerate(pbar):
            points = batch['points'].to(self.device, non_blocking=True)
            images = batch['images'].to(self.device, non_blocking=True)
            targets = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch['targets'].items()
            }
            aug_params = batch['aug_params']   # list[B]

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    output = self.model(points, images, aug_params)
                    loss, loss_dict = self.model.compute_loss(output, targets)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(points, images, aug_params)
                loss, loss_dict = self.model.compute_loss(output, targets)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

            self.loss_tracker.update(loss_dict)

            # progress bar
            batch_times.append(time.time() - t0)
            t0 = time.time()
            if batch_idx % self.print_interval == 0:
                avg_t = sum(batch_times[-20:]) / max(len(batch_times[-20:]), 1)
                mem   = torch.cuda.memory_allocated() / 1024**3 if self.device.type == 'cuda' else 0
                pbar.set_postfix({
                    'loss' : f'{loss.item():.3f}',
                    'mem'  : f'{mem:.1f}GB',
                    's/it' : f'{avg_t:.2f}'
                })

        return self.loss_tracker.get_average()

    # ── validation ────────────────────────────────────────────────────────────
    def validate(self) -> dict:
        self.model.eval()
        self.loss_tracker.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                points = batch['points'].to(self.device, non_blocking=True)
                images = batch['images'].to(self.device, non_blocking=True)
                targets = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch['targets'].items()
                }

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        output = self.model(points, images, aug_params=None)
                        _, loss_dict = self.model.compute_loss(output, targets)
                else:
                    output = self.model(points, images, aug_params=None)
                    _, loss_dict = self.model.compute_loss(output, targets)

                self.loss_tracker.update(loss_dict)

        return self.loss_tracker.get_average()

    # ── main loop ─────────────────────────────────────────────────────────────
    def train(self):
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        epochs    = self.config['training']['epochs']
        save_freq = self.config.get('logging', {}).get('save_freq', 5)
        start     = time.time()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            t_ep    = time.time()
            train_m = self.train_epoch(epoch, epochs)
            val_m   = self.validate()
            ep_min  = (time.time() - t_ep) / 60

            # scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_m['total'])
                else:
                    self.scheduler.step()

            elapsed = time.time() - start
            eta_h   = (elapsed / epoch) * (epochs - epoch) / 3600
            gpu_info = get_gpu_util()

            print(f"\nEpoch {epoch}/{epochs}  ({ep_min:.1f} min/epoch)")
            print(f"  Train Loss : {train_m.get('total', 0):.4f}")
            print(f"  Val Loss   : {val_m.get('total',   0):.4f}")
            print(f"  LR         : {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  ETA        : {eta_h:.1f} h")
            if gpu_info:
                print(f"  {gpu_info}")

            is_best = val_m.get('total', float('inf')) < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m['total']

            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            if self.early_stopping(val_m.get('total', float('inf'))):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        total_h = (time.time() - start) / 3600
        print(f"\nDone!  {total_h:.2f} h total  |  Best val loss: {self.best_val_loss:.4f}")

    # ── checkpoint ────────────────────────────────────────────────────────────
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        # torch.compile wraps model di _orig_mod — ambil model asli untuk state_dict
        raw = getattr(self.model, '_orig_mod', self.model)
        ckpt = {
            'epoch':            epoch,
            'model_state_dict': raw.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'best_val_loss':    self.best_val_loss,
            'config':           self.config
        }
        torch.save(ckpt, self.checkpoint_dir / 'latest.pth.tar')
        torch.save(ckpt, self.checkpoint_dir / f'epoch_{epoch}.pth.tar')
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'best.pth.tar')
            print(f"  ✓ Best checkpoint (val_loss={self.best_val_loss:.4f})")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='./config_rtx4080.yaml')
    p.add_argument('--data_path',  required=True)
    p.add_argument('--output_dir', default='./results')
    p.add_argument('--resume',     default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    trainer = Trainer(config, args)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw  = getattr(trainer.model, '_orig_mod', trainer.model)
        raw.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optim_state_dict'])
        trainer.current_epoch = ckpt['epoch']
        trainer.best_val_loss = ckpt['best_val_loss']
        print(f"Resumed from epoch {trainer.current_epoch}")

    trainer.train()


if __name__ == '__main__':
    main()