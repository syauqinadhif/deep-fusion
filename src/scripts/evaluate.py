#!/usr/bin/env python3
"""
Evaluation script for DeepFusion 3D Object Detection.

Usage:
    python evaluate.py --config ../config.yaml --checkpoint ./checkpoints/best.pth.tar --data_path /path/to/kitti
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_deepfusion_model, DeepFusionLite
from datasets import KITTIDataset, get_val_transforms, collate_fn
from utils import load_config, get_device, DetectionMetrics, Visualizer


class Evaluator:
    """
    Evaluator class for DeepFusion model.
    """

    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.device = get_device()

        # Build model
        print("Building model...")
        self.model = self.build_model()
        self.model.to(self.device)

        # Load checkpoint
        if args.checkpoint:
            self.load_checkpoint(args.checkpoint)

        # Build dataset
        print("Building validation dataset...")
        val_transforms = get_val_transforms(self.config)
        self.dataset = KITTIDataset(
            root_path=args.data_path,
            split='val',
            transform=val_transforms,
            class_names=['Car', 'Pedestrian', 'Cyclist']
        )

        # Build data loader
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )

        # Metrics
        self.metrics = DetectionMetrics(
            num_classes=3,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            iou_thresholds=config.get('evaluation', {}).get('iou_thresholds', [0.5, 0.7]),
            difficulty_levels=config.get('evaluation', {}).get('difficulty_levels', ['easy', 'moderate', 'hard'])
        )

        # Visualizer
        self.visualizer = Visualizer(class_names=['Car', 'Pedestrian', 'Cyclist'])

        # Output directory
        self.output_dir = Path(args.output_dir) if args.output_dir else Path('./results/evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_model(self):
        """Build model from config."""
        if args.model_type == 'lite':
            return DeepFusionLite(
                num_classes=self.config['model']['num_classes'],
                max_objects=self.config['model']['max_objects_per_image']
            )
        else:
            return build_deepfusion_model(self.config)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    def run_inference(self, save_predictions: bool = True):
        """Run inference on validation set."""
        print(f"\nRunning inference on {len(self.dataset)} samples...")

        self.model.eval()
        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.data_loader, desc="Inference")):
                # Move data to device
                points = batch['points'].to(self.device)
                images = batch['images'].to(self.device)

                # Run inference
                detections = self.model.inference(
                    points,
                    images,
                    conf_threshold=self.config.get('evaluation', {}).get('vis_threshold', 0.3),
                    nms_threshold=0.5
                )

                # Collect predictions
                for det in detections:
                    all_predictions.append(det)

                # Collect ground truths
                for i in range(len(batch['indices'])):
                    gt = {
                        'boxes': batch['targets']['size'][i],  # Placeholder
                        'labels': torch.randint(0, 3, (5,)),    # Placeholder
                        'difficulty': torch.zeros(5, dtype=torch.long)
                    }
                    all_ground_truths.append(gt)

                # Save visualizations
                if save_predictions and idx % 10 == 0:
                    self.save_visualization(batch, detections, idx)

        return all_predictions, all_ground_truths

    def save_visualization(self, batch, detections, idx):
        """Save visualization for a sample."""
        # Get first sample
        points = batch['points'][0].cpu().numpy()
        sample_idx = batch['indices'][0]

        if len(detections) > 0:
            det = detections[0]

            # Visualize BEV
            bev_img = self.visualizer.visualize_bev(
                points,
                det['boxes'].cpu().numpy() if torch.is_tensor(det['boxes']) else det['boxes'],
                det['labels'].cpu().numpy() if torch.is_tensor(det['labels']) else det['labels'],
                det['scores'].cpu().numpy() if torch.is_tensor(det['scores']) else det['scores'],
                save_path=str(self.output_dir / f'bev_{sample_idx}.jpg')
            )

    def compute_metrics(self, predictions, ground_truths):
        """Compute detection metrics."""
        print("\nComputing metrics...")

        # Update metrics for each difficulty level
        for difficulty in ['easy', 'moderate', 'hard']:
            self.metrics.update(predictions, ground_truths, difficulty=difficulty)

        # Compute results
        results = self.metrics.compute()

        # Print results
        print(f"\n{'='*60}")
        print("Detection Results (KITTI Style)")
        print(f"{'='*60}")

        for key, value in results.items():
            if key == 'mean':
                print(f"\nMean Performance:")
                print(f"  AP:       {value['ap']:.4f}")
                print(f"  Precision: {value['precision']:.4f}")
                print(f"  Recall:    {value['recall']:.4f}")
                print(f"  F1:        {value['f1']:.4f}")
            elif isinstance(value, dict):
                print(f"\n{key}:")
                print(f"  AP:       {value['ap']:.4f}")
                print(f"  Precision: {value['precision']:.4f}")
                print(f"  Recall:    {value['recall']:.4f}")
                print(f"  F1:        {value['f1']:.4f}")

        print(f"{'='*60}\n")

        # Save results
        results_file = self.output_dir / 'results.json'
        with open(results_file, 'w') as f:
            # Convert to serializable format
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {k: float(v) for k, v in value.items()}
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {results_file}")

        return results

    def evaluate(self):
        """Run complete evaluation."""
        # Run inference
        predictions, ground_truths = self.run_inference(
            save_predictions=self.args.save_predictions
        )

        # Compute metrics
        results = self.compute_metrics(predictions, ground_truths)

        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DeepFusion model')

    parser.add_argument(
        '--config',
        type=str,
        default='./config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
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
        default='./results/evaluation',
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
        '--save_predictions',
        action='store_true',
        help='Save prediction visualizations'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Create evaluator
    evaluator = Evaluator(config, args)

    # Run evaluation
    results = evaluator.evaluate()


if __name__ == '__main__':
    args = parse_args()
    main()
