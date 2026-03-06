#!/usr/bin/env python3
"""
Export script for DeepFusion model deployment.
Converts trained models to TorchScript and TensorRT for Jetson deployment.

Usage:
    python export.py --config ../config.yaml --checkpoint ./checkpoints/best.pth.tar --output_dir ./exported_models
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_deepfusion_model, DeepFusionLite
from utils import load_config, get_device


class ModelExporter:
    """
    Export DeepFusion model for deployment.
    """

    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.device = get_device()

        # Output directory
        self.output_dir = Path(args.output_dir) if args.output_dir else Path('./exported_models')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build model
        print("Building model...")
        self.model = self.build_model()
        self.model.to(self.device)

        # Load checkpoint
        if args.checkpoint:
            self.load_checkpoint(args.checkpoint)

        # Set model to eval mode
        self.model.eval()

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

        print("Checkpoint loaded successfully")

    def export_torchscript(self, example_inputs=None):
        """
        Export model to TorchScript format.

        Args:
            example_inputs: Example inputs for tracing (optional)
        """
        print("\n" + "="*60)
        print("Exporting to TorchScript")
        print("="*60)

        # Create example inputs if not provided
        if example_inputs is None:
            batch_size = 1
            num_points = 10000
            img_h, img_w = 384, 1280

            example_inputs = (
                torch.randn(batch_size, num_points, 4).to(self.device),
                torch.randn(batch_size, 3, img_h, img_w).to(self.device)
            )

        points, images = example_inputs

        # Try scripting first
        try:
            print("Attempting torch.script.compile...")
            scripted_model = torch.jit.script(self.model)
            print("✓ Scripting successful!")

            # Save scripted model
            scripted_path = self.output_dir / 'deepfusion_scripted.pt'
            scripted_model.save(str(scripted_path))
            print(f"Saved scripted model to {scripted_path}")

        except Exception as e:
            print(f"✗ Scripting failed: {e}")
            print("Attempting torch.jit.trace instead...")

            try:
                # Trace the model
                traced_model = torch.jit.trace(
                    self.model,
                    (points, images, None),
                    strict=False
                )
                print("✓ Tracing successful!")

                # Save traced model
                traced_path = self.output_dir / 'deepfusion_traced.pt'
                traced_model.save(str(traced_path))
                print(f"Saved traced model to {traced_path}")

            except Exception as e:
                print(f"✗ Tracing failed: {e}")
                return False

        return True

    def export_onnx(self, example_inputs=None):
        """
        Export model to ONNX format (for TensorRT conversion).

        Args:
            example_inputs: Example inputs for export (optional)
        """
        print("\n" + "="*60)
        print("Exporting to ONNX")
        print("="*60)

        try:
            # Create example inputs if not provided
            if example_inputs is None:
                batch_size = 1
                num_points = 10000
                img_h, img_w = 384, 1280

                example_inputs = (
                    torch.randn(batch_size, num_points, 4).to(self.device),
                    torch.randn(batch_size, 3, img_h, img_w).to(self.device)
                )

            points, images = example_inputs

            # Export to ONNX
            onnx_path = self.output_dir / 'deepfusion.onnx'

            torch.onnx.export(
                self.model,
                (points, images, None),  # aug_params is None
                str(onnx_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['points', 'images', 'aug_params'],
                output_names=['predictions'],
                dynamic_axes={
                    'points': {0: 'batch_size', 1: 'num_points'},
                    'images': {0: 'batch_size'},
                    'predictions': {0: 'batch_size'}
                }
            )

            print(f"✓ ONNX export successful!")
            print(f"Saved ONNX model to {onnx_path}")

            # Verify ONNX model
            print("\nVerifying ONNX model...")
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model verification passed!")

            return True

        except ImportError:
            print("✗ ONNX library not installed. Install with: pip install onnx onnxruntime")
            return False
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            return False

    def export_fp16(self):
        """
        Convert model to FP16 precision for faster inference.
        """
        print("\n" + "="*60)
        print("Converting to FP16")
        print("="*60)

        try:
            # Convert model to FP16
            fp16_model = self.model.half()

            # Save FP16 model checkpoint
            fp16_path = self.output_dir / 'deepfusion_fp16.pth'
            torch.save({
                'model_state_dict': fp16_model.state_dict(),
                'config': self.config
            }, fp16_path)

            print(f"✓ FP16 conversion successful!")
            print(f"Saved FP16 model to {fp16_path}")

            # Test FP16 model
            print("\nTesting FP16 model...")
            batch_size = 1
            num_points = 10000
            img_h, img_w = 384, 1280

            points_fp16 = torch.randn(batch_size, num_points, 4).half().to(self.device)
            images_fp16 = torch.randn(batch_size, 3, img_h, img_w).half().to(self.device)

            with torch.no_grad():
                output = fp16_model(points_fp16, images_fp16, None)

            print(f"✓ FP16 model test passed! Output shape: {output['predictions']['heatmap'].shape}")

            return True

        except Exception as e:
            print(f"✗ FP16 conversion failed: {e}")
            return False

    def prepare_for_tensorrt(self):
        """
        Prepare model for TensorRT conversion.
        This exports ONNX model which can be converted to TensorRT engine.
        """
        print("\n" + "="*60)
        print("Preparing for TensorRT")
        print("="*60)

        # Export ONNX model for TensorRT
        success = self.export_onnx()

        if success:
            print("\n✓ Model ready for TensorRT conversion!")
            print("\nTo convert to TensorRT engine, use trtexec on Jetson:")
            print(f"  trtexec --onnx={self.output_dir / 'deepfusion.onnx'} \\")
            print("           --saveEngine=deepfusion.trt \\")
            print("           --fp16 \\")
            print("           --workspace=4096")

            # Create conversion script
            script_path = self.output_dir / 'convert_to_tensorrt.sh'
            with open(script_path, 'w') as f:
                f.write("#!/bin/bash\n\n")
                f.write("# Convert ONNX model to TensorRT engine\n")
                f.write(f"trtexec --onnx=deepfusion.onnx \\\n")
                f.write("       --saveEngine=deepfusion.trt \\\n")
                f.write("       --fp16 \\\n")
                f.write("       --workspace=4096\n")

            print(f"\nConversion script saved to {script_path}")
            print(f"Run: bash {script_path}")

            return True
        else:
            return False

    def export_for_jetson(self):
        """
        Export model in optimal format for Jetson deployment.
        """
        print("\n" + "="*60)
        print("Exporting for Jetson Deployment")
        print("="*60)

        results = {}

        # Export TorchScript (most compatible)
        results['torchscript'] = self.export_torchscript()

        # Export FP16 model
        results['fp16'] = self.export_fp16()

        # Prepare for TensorRT (best performance)
        results['tensorrt'] = self.prepare_for_tensorrt()

        # Save model info
        info = {
            'model_type': self.args.model_type,
            'config': self.config,
            'exports': results
        }

        import json
        info_path = self.output_dir / 'export_info.json'
        with open(info_path, 'w') as f:
            json.dump({k: v for k, v in info.items() if k != 'config'}, f, indent=2)

        print(f"\nModel information saved to {info_path}")

        # Summary
        print("\n" + "="*60)
        print("Export Summary")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Exports created:")
        print(f"  - TorchScript:   {'✓' if results['torchscript'] else '✗'}")
        print(f"  - FP16 Model:    {'✓' if results['fp16'] else '✗'}")
        print(f"  - TensorRT Ready: {'✓' if results['tensorrt'] else '✗'}")
        print("="*60)

        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export DeepFusion model for deployment')

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
        '--output_dir',
        type=str,
        default='./exported_models',
        help='Output directory for exported models'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        choices=['standard', 'lite'],
        default='standard',
        help='Model type: standard or lite'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['torchscript', 'onnx', 'fp16', 'tensorrt', 'all'],
        default='all',
        help='Export format (default: all)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Create exporter
    exporter = ModelExporter(config, args)

    # Export based on format
    if args.format == 'torchscript':
        exporter.export_torchscript()
    elif args.format == 'onnx':
        exporter.export_onnx()
    elif args.format == 'fp16':
        exporter.export_fp16()
    elif args.format == 'tensorrt':
        exporter.prepare_for_tensorrt()
    else:  # 'all'
        exporter.export_for_jetson()


if __name__ == '__main__':
    args = parse_args()
    main()
