"""
Image encoder for extracting features from camera images.
Uses ResNet34 as backbone with pretrained weights.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class ImageEncoder(nn.Module):
    """
    Image encoder using ResNet backbone.

    Args:
        backbone: ResNet version ('resnet18', 'resnet34', 'resnet50')
        pretrained: Whether to use ImageNet pretrained weights
        out_features: Number of output features
        input_channels: Number of input channels (default: 3 for RGB)
    """

    def __init__(
        self,
        backbone: str = 'resnet34',
        pretrained: bool = True,
        out_features: int = 256,
        input_channels: int = 3
    ):
        super().__init__()

        self.backbone_name = backbone
        self.out_features = out_features

        # Load pretrained ResNet
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_channels = [64, 128, 256, 512]
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_channels = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Modify first conv if needed
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(
                input_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )

        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Projection layers to unify feature dimensions
        self.proj1 = nn.Conv2d(self.feature_channels[0], out_features, kernel_size=1)
        self.proj2 = nn.Conv2d(self.feature_channels[1], out_features, kernel_size=1)
        self.proj3 = nn.Conv2d(self.feature_channels[2], out_features, kernel_size=1)
        self.proj4 = nn.Conv2d(self.feature_channels[3], out_features, kernel_size=1)

        # Feature aggregation
        self.aggregation = nn.Sequential(
            nn.Conv2d(out_features * 4, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of image encoder.

        Args:
            images: (batch_size, 3, H, W) input images

        Returns:
            features: (batch_size, out_features, H', W') aggregated features
            feature_dict: Dictionary with multi-scale features
        """
        # Initial layers
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Extract multi-scale features
        feat1 = self.layer1(x)  # 1/4 resolution
        feat2 = self.layer2(feat1)  # 1/8 resolution
        feat3 = self.layer3(feat2)  # 1/16 resolution
        feat4 = self.layer4(feat3)  # 1/32 resolution

        # Project to same dimension
        feat1 = self.proj1(feat1)
        feat2 = self.proj2(feat2)
        feat3 = self.proj3(feat3)
        feat4 = self.proj4(feat4)

        # Upsample and concatenate
        feat2_up = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        feat3_up = F.interpolate(feat3, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        feat4_up = F.interpolate(feat4, size=feat1.shape[2:], mode='bilinear', align_corners=False)

        concat_feat = torch.cat([feat1, feat2_up, feat3_up, feat4_up], dim=1)

        # Aggregate
        features = self.aggregation(concat_feat)

        feature_dict = {
            'feat1': feat1,
            'feat2': feat2,
            'feat3': feat3,
            'feat4': feat4,
            'aggregated': features
        }

        return features, feature_dict


class LightweightImageEncoder(nn.Module):
    """
    Lightweight image encoder for faster inference on Jetson.

    Uses a smaller backbone and optimizations for edge deployment.
    """

    def __init__(
        self,
        out_features: int = 256,
        input_channels: int = 3,
        width_mult: float = 0.5  # Width multiplier for MobileNetV2
    ):
        super().__init__()

        self.out_features = out_features

        # Use MobileNetV2 for efficiency
        mobilenet = models.mobilenet_v2(pretrained=True)

        # Modify first conv if needed
        if input_channels != 3:
            mobilenet.features[0][0] = nn.Conv2d(
                input_channels, 32,
                kernel_size=3, stride=2, padding=1, bias=False
            )

        # Extract features up to layer 12 (good balance)
        self.features = mobilenet.features[:14]

        # Get feature channels from MobileNetV2
        self.in_channels = 96  # Output of layer 13 in MobileNetV2

        # Projection
        self.projection = nn.Sequential(
            nn.Conv2d(self.in_channels, out_features, kernel_size=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU6(inplace=True)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of lightweight image encoder.

        Args:
            images: (batch_size, 3, H, W) input images

        Returns:
            features: (batch_size, out_features, H', W') features
        """
        x = self.features(images)
        features = self.projection(x)
        return features


import torch.nn.functional as F


if __name__ == "__main__":
    # Test the image encoder
    batch_size = 2

    # Create dummy images (typical KITTI size: ~375x1242)
    images = torch.randn(batch_size, 3, 384, 1280)

    # Test ResNet34 encoder
    print("Testing ResNet34 encoder...")
    model = ImageEncoder(
        backbone='resnet34',
        pretrained=False,
        out_features=256
    )

    features, feature_dict = model(images)

    print(f"Input shape: {images.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Multi-scale features:")
    for name, feat in feature_dict.items():
        print(f"  {name}: {feat.shape}")

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test lightweight encoder
    print("\nTesting Lightweight encoder...")
    light_model = LightweightImageEncoder(out_features=256)
    light_features = light_model(images)

    print(f"Input shape: {images.shape}")
    print(f"Output features shape: {light_features.shape}")
    print(f"Total parameters: {sum(p.numel() for p in light_model.parameters()):,}")
