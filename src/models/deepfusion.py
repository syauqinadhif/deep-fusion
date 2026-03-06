"""
DeepFusion Model - Main integration of all components.
Lidar-Camera fusion for 3D object detection (Detection Only).

This is the main model that integrates:
1. PointPillars backbone for LiDAR features
2. Image encoder for camera features
3. Inverse Augmentation for geometric alignment
4. Learnable Alignment for feature fusion
5. Detection Head for 3D bounding box prediction
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .pointpillars import PointPillarsBackbone
from .image_encoder import ImageEncoder
from .inverse_aug import InverseAugmentation, AugmentationParams
from .learnable_align import LearnableAlignment
from .detection_head import DetectionHead, ObjectDetectionLoss


class DeepFusion(nn.Module):
    """
    DeepFusion model for 3D object detection.

    Args:
        lidar_channels: LiDAR backbone output channels
        image_channels: Image encoder output channels
        hidden_dim: Hidden dimension for alignment
        num_heads: Number of attention heads
        num_layers: Number of alignment layers
        num_classes: Number of object classes
        max_objects: Maximum number of objects per image
    """

    def __init__(
        self,
        lidar_channels: int = 256,
        image_channels: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        num_classes: int = 3,
        max_objects: int = 512,
        image_backbone: str = 'resnet34',
        pretrained_image: bool = True,
        # PointPillars config
        max_points_per_pillar: int = 100,
        max_pillars: int = 12000,
        voxel_size: list = [0.16, 0.16, 4.0],
        point_range: list = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        super().__init__()

        self.num_classes = num_classes
        self.max_objects = max_objects
        self.hidden_dim = hidden_dim

        # LiDAR backbone (PointPillars)
        self.lidar_backbone = PointPillarsBackbone(
            in_channels=4,
            out_channels=lidar_channels,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
            voxel_size=voxel_size,
            point_range=point_range
        )

        # Image encoder
        self.image_encoder = ImageEncoder(
            backbone=image_backbone,
            pretrained=pretrained_image,
            out_features=image_channels
        )

        # Inverse augmentation module
        self.inverse_aug = InverseAugmentation()

        # Learnable alignment module
        self.learnable_align = LearnableAlignment(
            lidar_channels=lidar_channels,
            image_channels=image_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        # Detection head
        self.detection_head = DetectionHead(
            in_channels=hidden_dim,
            num_classes=num_classes,
            max_objects=max_objects
        )

        # Loss function
        self.loss_fn = ObjectDetectionLoss(
            num_classes=num_classes,
            alpha=0.25,
            beta=2.0,
            gamma=2.0,
            offset_weight=1.0,
            size_weight=1.0,
            rotation_weight=1.0,
            z_weight=1.0
        )

    def forward(
        self,
        points: torch.Tensor,
        images: torch.Tensor,
        aug_params: Optional[AugmentationParams] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of DeepFusion.

        Args:
            points: (B, N, 4) LiDAR point cloud [x, y, z, intensity]
            images: (B, 3, H, W) camera images
            aug_params: Augmentation parameters (optional)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - predictions: Model predictions (heatmap, offset, size, rotation, z)
                - attention_weights: Attention weights (optional)
                - lidar_features: Intermediate LiDAR features (for visualization)
                - image_features: Intermediate image features (for visualization)
                - aligned_features: Aligned/fused features (for visualization)
        """
        # Extract LiDAR features
        lidar_features = self.lidar_backbone(points)  # (B, C, H, W)

        # Extract image features
        image_features, image_feat_dict = self.image_encoder(images)  # (B, C, H', W')

        # Resize image features to match LiDAR feature resolution
        if image_features.shape[2:] != lidar_features.shape[2:]:
            image_features = torch.nn.functional.interpolate(
                image_features,
                size=lidar_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Apply inverse augmentation for geometric alignment
        aligned_lidar, aligned_image = self.inverse_aug(
            lidar_features,
            image_features,
            aug_params
        )

        # Apply learnable alignment for feature fusion
        fused_features, attention_weights = self.learnable_align(
            aligned_lidar,
            aligned_image,
            return_attention=return_attention
        )

        # Detection head predictions
        predictions = self.detection_head(fused_features)

        output = {
            'predictions': predictions,
            'lidar_features': lidar_features,
            'image_features': image_features,
            'fused_features': fused_features
        }

        if return_attention:
            output['attention_weights'] = attention_weights

        return output

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute detection loss.

        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with ground truth targets

        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with individual loss components
        """
        return self.loss_fn(predictions['predictions'], targets)

    def inference(
        self,
        points: torch.Tensor,
        images: torch.Tensor,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.5
    ) -> list:
        """
        Run inference and return decoded detections.

        Args:
            points: (B, N, 4) LiDAR point cloud
            images: (B, 3, H, W) camera images
            conf_threshold: Confidence threshold for detections
            nms_threshold: IoU threshold for NMS

        Returns:
            List of detections per batch element
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            output = self.forward(points, images)

            # Decode predictions
            from .detection_head import DetectionDecoder
            decoder = DetectionDecoder(
                num_classes=self.num_classes,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                max_objects=self.max_objects
            )

            detections = decoder.decode(output['predictions'])

        return detections

    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'hidden_dim': self.hidden_dim,
            'max_objects': self.max_objects
        }


class DeepFusionLite(nn.Module):
    """
    Lightweight version of DeepFusion for Jetson deployment.

    Optimizations:
    - Smaller image backbone (MobileNetV2 instead of ResNet)
    - Fewer alignment layers
    - Reduced feature dimensions
    """

    def __init__(
        self,
        lidar_channels: int = 128,
        image_channels: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 3,
        max_objects: int = 512,
        # PointPillars config
        max_points_per_pillar: int = 100,
        max_pillars: int = 12000,
        voxel_size: list = [0.16, 0.16, 4.0],
        point_range: list = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        super().__init__()

        self.num_classes = num_classes
        self.max_objects = max_objects
        self.hidden_dim = hidden_dim

        # LiDAR backbone
        self.lidar_backbone = PointPillarsBackbone(
            in_channels=4,
            out_channels=lidar_channels,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
            voxel_size=voxel_size,
            point_range=point_range
        )

        # Lightweight image encoder
        from .image_encoder import LightweightImageEncoder
        self.image_encoder = LightweightImageEncoder(
            out_features=image_channels
        )

        # Inverse augmentation
        self.inverse_aug = InverseAugmentation()

        # Single-layer learnable alignment
        self.learnable_align = LearnableAlignment(
            lidar_channels=lidar_channels,
            image_channels=image_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=1  # Single layer for efficiency
        )

        # Detection head
        self.detection_head = DetectionHead(
            in_channels=hidden_dim,
            num_classes=num_classes,
            max_objects=max_objects
        )

        self.loss_fn = ObjectDetectionLoss(num_classes=num_classes)

    def forward(
        self,
        points: torch.Tensor,
        images: torch.Tensor,
        aug_params: Optional[AugmentationParams] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of DeepFusion Lite."""
        # Extract LiDAR features
        lidar_features = self.lidar_backbone(points)

        # Extract image features
        image_features = self.image_encoder(images)

        # Resize to match
        if image_features.shape[2:] != lidar_features.shape[2:]:
            image_features = torch.nn.functional.interpolate(
                image_features,
                size=lidar_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Inverse augmentation
        aligned_lidar, aligned_image = self.inverse_aug(
            lidar_features,
            image_features,
            aug_params
        )

        # Learnable alignment
        fused_features, _ = self.learnable_align(aligned_lidar, aligned_image)

        # Detection
        predictions = self.detection_head(fused_features)

        return {
            'predictions': predictions,
            'lidar_features': lidar_features,
            'image_features': image_features,
            'fused_features': fused_features
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss."""
        return self.loss_fn(predictions['predictions'], targets)


def build_deepfusion_model(config: dict, lite: bool = False) -> nn.Module:
    """
    Build DeepFusion model from configuration.

    Args:
        config: Configuration dictionary
        lite: Whether to build lite version

    Returns:
        DeepFusion model
    """
    model_config = config.get('model', {})

    if lite:
        model = DeepFusionLite(
            lidar_channels=model_config.get('image_features', 128),
            image_channels=model_config.get('image_features', 128),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_heads=model_config.get('n_heads', 4),
            num_classes=model_config.get('num_classes', 3),
            max_objects=model_config.get('max_objects_per_image', 512),
            max_points_per_pillar=model_config.get('max_points_per_pillar', 100),
            max_pillars=model_config.get('max_pillars', 12000),
            voxel_size=model_config.get('voxel_size', [0.16, 0.16, 4.0]),
            point_range=model_config.get('point_range', [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0])
        )
    else:
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


if __name__ == "__main__":
    # Test the DeepFusion model
    batch_size = 2
    num_points = 1000

    # Create dummy inputs
    points = torch.randn(batch_size, num_points, 4)
    points[:, :, :3] *= 10  # Scale coordinates
    points[:, :, 3] = torch.abs(points[:, :, 3])  # Positive intensity

    images = torch.randn(batch_size, 3, 384, 1280)

    # Test standard DeepFusion
    print("Testing DeepFusion (standard)...")
    model = DeepFusion(
        lidar_channels=256,
        image_channels=256,
        hidden_dim=256,
        num_heads=8,
        num_layers=1,
        num_classes=3
    )

    output = model(points, images, return_attention=True)

    print(f"Input points shape: {points.shape}")
    print(f"Input images shape: {images.shape}")
    print(f"Output keys: {output.keys()}")
    print(f"Predictions:")
    for name, pred in output['predictions'].items():
        print(f"  {name}: {pred.shape}")

    model_info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Total parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")

    # Test DeepFusion Lite
    print("\n" + "="*60)
    print("Testing DeepFusion Lite...")
    lite_model = DeepFusionLite(
        lidar_channels=128,
        image_channels=128,
        hidden_dim=128,
        num_heads=4,
        num_classes=3
    )

    lite_output = lite_model(points, images)

    print(f"Predictions:")
    for name, pred in lite_output['predictions'].items():
        print(f"  {name}: {pred.shape}")

    total_params = sum(p.numel() for p in lite_model.parameters())
    print(f"\nLite Model Total parameters: {total_params:,}")
    print(f"Parameter reduction: {(1 - total_params/model_info['total_parameters'])*100:.1f}%")
