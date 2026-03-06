#!/usr/bin/env python3
"""
Jetson AGX Orin deployment script for DeepFusion.
Real-time 3D object detection on edge device.

Usage:
    python jetson_deploy.py --model_path ./exported_models/deepfusion_fp16.pth --data_path /path/to/kitti
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DeepFusionLite
from utils import load_config, get_device, Visualizer


class JetsonInferencer:
    """
    Real-time inference engine for Jetson AGX Orin.
    """

    def __init__(self, model_path: str, args):
        self.args = args
        self.device = get_device()

        print(f"Running on device: {self.device}")
        print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        # Load model
        print(f"\nLoading model from {model_path}...")
        self.model = self.load_model(model_path)
        self.model.eval()

        # Optimization settings
        if torch.cuda.is_available():
            self.setup_optimizations()

        # Timing
        self.inference_times = []

        # Visualizer
        self.visualizer = Visualizer()

    def load_model(self, model_path: str) -> nn.Module:
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load config if available
        config = checkpoint.get('config', {})

        # Build model
        if config:
            model = DeepFusionLite(
                num_classes=config.get('model', {}).get('num_classes', 3),
                max_objects=config.get('model', {}).get('max_objects_per_image', 512)
            )
        else:
            # Default configuration
            model = DeepFusionLite(
                num_classes=3,
                max_objects=512
            )

        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)

        print("Model loaded successfully!")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def setup_optimizations(self):
        """Setup Jetson-specific optimizations."""
        print("\nSetting up Jetson optimizations...")

        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True

        # Disable gradient calculation for inference
        torch.set_grad_enabled(False)

        # Set TensorRT flags if available
        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print("✓ cuDNN benchmark enabled")
        print("✓ Gradient calculation disabled")

        # Check if FP16 is supported
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
            print("✓ FP16 inference supported")
            self.fp16_supported = True
        else:
            print("✗ FP16 not supported, using FP32")
            self.fp16_supported = False

    def preprocess(
        self,
        points: np.ndarray,
        image: np.ndarray
    ) -> tuple:
        """Preprocess input data."""
        # Convert point cloud to tensor
        points_tensor = torch.from_numpy(points).float().unsqueeze(0)  # (1, N, 4)

        # Convert image to tensor
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        image_tensor = torch.from_numpy(image).float() / 255.0
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        return points_tensor, image_tensor

    def inference(
        self,
        points: np.ndarray,
        image: np.ndarray,
        conf_threshold: float = 0.3
    ) -> dict:
        """
        Run inference on a single sample.

        Args:
            points: (N, 4) point cloud
            image: (H, W, 3) RGB image
            conf_threshold: Confidence threshold for detections

        Returns:
            Dictionary with detection results
        """
        # Preprocess
        points_tensor, image_tensor = self.preprocess(points, image)
        points_tensor = points_tensor.to(self.device)
        image_tensor = image_tensor.to(self.device)

        # Convert to FP16 if supported
        if self.fp16_supported and self.args.fp16:
            points_tensor = points_tensor.half()
            image_tensor = image_tensor.half()

        # Start timing
        start_time = time.perf_counter()

        # Run inference
        with torch.no_grad():
            detections = self.model.inference(
                points_tensor,
                image_tensor,
                conf_threshold=conf_threshold,
                nms_threshold=0.5
            )

        # End timing
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        self.inference_times.append(inference_time)

        # Process detections
        if len(detections) > 0:
            det = detections[0]

            # Move to CPU and convert to numpy
            results = {
                'boxes': det['boxes'].cpu().numpy(),
                'labels': det['labels'].cpu().numpy(),
                'scores': det['scores'].cpu().numpy(),
                'num_detections': len(det['scores']),
                'inference_time_ms': inference_time
            }
        else:
            results = {
                'boxes': np.empty((0, 7)),
                'labels': np.empty((0,), dtype=np.int64),
                'scores': np.empty((0,)),
                'num_detections': 0,
                'inference_time_ms': inference_time
            }

        return results

    def benchmark(self, num_iterations: int = 100):
        """Run benchmark to measure performance."""
        print(f"\n{'='*60}")
        print("Running Benchmark")
        print(f"{'='*60}")

        # Create dummy data
        num_points = 10000
        img_h, img_w = 384, 1280

        points = np.random.randn(num_points, 4).astype(np.float32)
        points[:, :3] *= 10
        points[:, 3] = np.abs(points[:, 3])

        image = np.random.rand(img_h, img_w, 3).astype(np.float32)

        print(f"Input: {num_points} points, {img_h}x{img_w} image")
        print(f"Running {num_iterations} iterations...")

        # Warmup
        print("Warming up...")
        for _ in range(10):
            _ = self.inference(points, image)

        # Benchmark
        print("Benchmarking...")
        self.inference_times = []

        for i in range(num_iterations):
            results = self.inference(points, image)

            if (i + 1) % 20 == 0:
                avg_time = np.mean(self.inference_times)
                fps = 1000 / avg_time
                print(f"  Iteration {i+1}/{num_iterations}: {avg_time:.2f}ms ({fps:.1f} FPS)")

        # Compute statistics
        times = np.array(self.inference_times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        median_time = np.median(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)

        fps = 1000 / mean_time

        print(f"\n{'='*60}")
        print("Benchmark Results")
        print(f"{'='*60}")
        print(f"Mean latency:     {mean_time:.2f} ms")
        print(f"Std deviation:    {std_time:.2f} ms")
        print(f"Min latency:      {min_time:.2f} ms")
        print(f"Max latency:      {max_time:.2f} ms")
        print(f"Median latency:   {median_time:.2f} ms")
        print(f"P95 latency:      {p95_time:.2f} ms")
        print(f"P99 latency:      {p99_time:.2f} ms")
        print(f"{'='*60}")
        print(f"Average FPS:      {fps:.2f}")
        print(f"{'='*60}")

        # Check if target is met
        target_fps = 30
        if fps >= target_fps:
            print(f"\n✓ Target FPS ({target_fps}) MET! Ready for real-time deployment.")
        else:
            print(f"\n✗ Target FPS ({target_fps}) NOT MET. Current: {fps:.2f} FPS")
            print("  Consider:")
            print("  - Using FP16 precision")
            print("  - Converting to TensorRT")
            print("  - Using DeepFusion Lite model")

        return {
            'mean_latency_ms': mean_time,
            'std_latency_ms': std_time,
            'fps': fps,
            'target_met': fps >= target_fps
        }

    def run_interactive(self):
        """Run interactive inference loop."""
        print(f"\n{'='*60}")
        print("Interactive Inference Mode")
        print(f"{'='*60}")
        print("Press 'q' to quit, 's' to save visualization")
        print()

        # This would typically connect to a real sensor
        # For demonstration, use dummy data
        try:
            while True:
                # Simulate new data
                num_points = 10000
                img_h, img_w = 384, 1280

                points = np.random.randn(num_points, 4).astype(np.float32)
                points[:, :3] *= 10
                points[:, 3] = np.abs(points[:, 3])

                image = np.random.rand(img_h, img_w, 3).astype(np.float32)

                # Run inference
                results = self.inference(points, image)

                # Print results
                print(f"\rDetections: {results['num_detections']} | "
                      f"Latency: {results['inference_time_ms']:.2f}ms | "
                      f"FPS: {1000/results['inference_time_ms']:.1f}", end='')

                # In real deployment, this would wait for sensor data
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nExiting...")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Jetson deployment for DeepFusion')

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['benchmark', 'interactive'],
        default='benchmark',
        help='Running mode'
    )

    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 precision for inference'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create inferencer
    inferencer = JetsonInferencer(args.model_path, args)

    # Run based on mode
    if args.mode == 'benchmark':
        results = inferencer.benchmark(args.iterations)

        # Save results
        output_path = Path('./jetson_benchmark_results.json')
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to {output_path}")

    else:  # interactive
        inferencer.run_interactive()


if __name__ == '__main__':
    args = parse_args()
    main()
